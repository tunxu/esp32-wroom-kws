// ESP-IDF + INMP441 + TensorFlow Lite Micro example
// Records audio via I2S, computes MelSpectrogram-like features, then runs inference.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "driver/i2s.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_check.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "model.h"  // generated from kws_DSCNN_model.tflite

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr const char* kTag = "kws_inmp441";

// ---- Audio capture config (INMP441) ----
constexpr i2s_port_t kI2sPort = I2S_NUM_0;
constexpr int kSampleRateHz = 16000;
constexpr int kRecordMs = 1000;  // 1 second window
constexpr int kNumSamples = (kSampleRateHz * kRecordMs) / 1000;

// Set these pins to your wiring.
constexpr gpio_num_t kPinBck = GPIO_NUM_26;   // SCK/BCLK
constexpr gpio_num_t kPinWs = GPIO_NUM_32;    // WS/LRCLK
constexpr gpio_num_t kPinData = GPIO_NUM_33;  // SD

// ---- Feature config (aligned to training scripts) ----
constexpr int kNfft = 1024;
constexpr int kHop = 512;
constexpr int kNMels = 64;
constexpr int kNFrames = 32;  // with centered padding: 1 + 16000/512 = 32
constexpr int kFftBins = (kNfft / 2) + 1;

// ---- TFLM config ----
constexpr size_t kTensorArenaSize = 100 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

int16_t audio_pcm[kNumSamples];
float mel_features[kNMels * kNFrames];
float hann_window[kNfft];
float mel_filterbank[kNMels][kFftBins];
bool frontend_ready = false;
}  // namespace

static float hz_to_mel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static void init_frontend_tables() {
  // Hann window
  for (int n = 0; n < kNfft; ++n) {
    hann_window[n] = 0.5f - 0.5f * cosf((2.0f * static_cast<float>(M_PI) * n) / (kNfft - 1));
  }

  // HTK-style mel filterbank, triangular, no normalization (similar to torchaudio defaults).
  memset(mel_filterbank, 0, sizeof(mel_filterbank));
  const float fmin = 0.0f;
  const float fmax = static_cast<float>(kSampleRateHz) / 2.0f;
  const float mel_min = hz_to_mel(fmin);
  const float mel_max = hz_to_mel(fmax);

  std::vector<float> mel_points(kNMels + 2);
  std::vector<int> bin_points(kNMels + 2);

  for (int i = 0; i < kNMels + 2; ++i) {
    float mel = mel_min + (mel_max - mel_min) * (static_cast<float>(i) / (kNMels + 1));
    float hz = mel_to_hz(mel);
    int bin = static_cast<int>(floorf((kNfft + 1) * hz / kSampleRateHz));
    if (bin < 0) bin = 0;
    if (bin > kFftBins - 1) bin = kFftBins - 1;
    mel_points[i] = mel;
    bin_points[i] = bin;
  }

  for (int m = 1; m <= kNMels; ++m) {
    int left = bin_points[m - 1];
    int center = bin_points[m];
    int right = bin_points[m + 1];

    if (center <= left) center = left + 1;
    if (right <= center) right = center + 1;
    if (right > kFftBins - 1) right = kFftBins - 1;

    for (int k = left; k < center && k < kFftBins; ++k) {
      mel_filterbank[m - 1][k] = (k - left) / static_cast<float>(center - left);
    }
    for (int k = center; k <= right && k < kFftBins; ++k) {
      mel_filterbank[m - 1][k] = (right - k) / static_cast<float>(right - center);
    }
  }

  frontend_ready = true;
}

static esp_err_t init_i2s_mic() {
  const i2s_config_t i2s_config = {
      .mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = kSampleRateHz,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 256,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0,
  };

  const i2s_pin_config_t pin_config = {
      .bck_io_num = kPinBck,
      .ws_io_num = kPinWs,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = kPinData,
  };

  ESP_RETURN_ON_ERROR(i2s_driver_install(kI2sPort, &i2s_config, 0, nullptr), kTag,
                      "i2s_driver_install failed");
  ESP_RETURN_ON_ERROR(i2s_set_pin(kI2sPort, &pin_config), kTag, "i2s_set_pin failed");
  ESP_RETURN_ON_ERROR(i2s_zero_dma_buffer(kI2sPort), kTag, "i2s_zero_dma_buffer failed");
  return ESP_OK;
}

static bool record_audio_1s(int16_t* dst, int samples) {
  int written = 0;
  int32_t i2s_raw[256];  // INMP441: 24-bit in 32-bit frame.

  while (written < samples) {
    size_t bytes_read = 0;
    esp_err_t err = i2s_read(kI2sPort, i2s_raw, sizeof(i2s_raw), &bytes_read,
                             pdMS_TO_TICKS(1000));
    if (err != ESP_OK || bytes_read == 0) {
      ESP_LOGE(kTag, "i2s_read failed: %s", esp_err_to_name(err));
      return false;
    }

    int got = bytes_read / sizeof(int32_t);
    for (int i = 0; i < got && written < samples; ++i) {
      dst[written++] = static_cast<int16_t>(i2s_raw[i] >> 14);
    }
  }
  return true;
}

static void pcm_to_mel_features(const int16_t* pcm, float* out_mel) {
  // 1) Convert to float and normalize waveform by max(abs(.)) like training script.
  float wav[kNumSamples];
  float max_abs = 1e-9f;
  for (int i = 0; i < kNumSamples; ++i) {
    wav[i] = static_cast<float>(pcm[i]) / 32768.0f;
    float a = fabsf(wav[i]);
    if (a > max_abs) max_abs = a;
  }
  for (int i = 0; i < kNumSamples; ++i) {
    wav[i] /= max_abs;
  }

  // 2) Center padding to mimic torchaudio MelSpectrogram(center=True).
  constexpr int kPad = kNfft / 2;  // 512
  float padded[kNumSamples + 2 * kPad];
  memset(padded, 0, sizeof(padded));
  for (int i = 0; i < kNumSamples; ++i) {
    padded[i + kPad] = wav[i];
  }

  // 3) STFT power spectrum + mel projection.
  float frame[kNfft];
  float power[kFftBins];

  for (int t = 0; t < kNFrames; ++t) {
    const int start = t * kHop;
    for (int n = 0; n < kNfft; ++n) {
      frame[n] = padded[start + n] * hann_window[n];
    }

    // Naive DFT (simple and dependency-free example).
    for (int k = 0; k < kFftBins; ++k) {
      float re = 0.0f;
      float im = 0.0f;
      const float coeff = -2.0f * static_cast<float>(M_PI) * k / kNfft;
      for (int n = 0; n < kNfft; ++n) {
        float phase = coeff * n;
        re += frame[n] * cosf(phase);
        im += frame[n] * sinf(phase);
      }
      power[k] = (re * re) + (im * im);  // power=2 like torchaudio default.
    }

    for (int m = 0; m < kNMels; ++m) {
      float e = 0.0f;
      for (int k = 0; k < kFftBins; ++k) {
        e += mel_filterbank[m][k] * power[k];
      }
      out_mel[m * kNFrames + t] = e;
    }
  }
}

static void fill_model_input_from_mel(TfLiteTensor* input, const float* mel) {
  const int expected = kNMels * kNFrames;
  int elems = 0;

  if (input->type == kTfLiteInt8) {
    elems = input->bytes;
    int copy_n = (expected < elems) ? expected : elems;
    for (int i = 0; i < copy_n; ++i) {
      int32_t q = static_cast<int32_t>(roundf(mel[i] / input->params.scale)) +
                  input->params.zero_point;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      input->data.int8[i] = static_cast<int8_t>(q);
    }
    for (int i = copy_n; i < elems; ++i) input->data.int8[i] = input->params.zero_point;
  } else if (input->type == kTfLiteFloat32) {
    elems = input->bytes / static_cast<int>(sizeof(float));
    int copy_n = (expected < elems) ? expected : elems;
    for (int i = 0; i < copy_n; ++i) input->data.f[i] = mel[i];
    for (int i = copy_n; i < elems; ++i) input->data.f[i] = 0.0f;
  } else {
    ESP_LOGE(kTag, "Unsupported input tensor type: %d", input->type);
    return;
  }

  if (elems != expected) {
    ESP_LOGW(kTag, "Model input elements=%d, mel elements=%d (truncated/padded).", elems,
             expected);
  }
}

static int argmax_int8(const int8_t* data, int n) {
  int idx = 0;
  int8_t best = data[0];
  for (int i = 1; i < n; ++i) {
    if (data[i] > best) {
      best = data[i];
      idx = i;
    }
  }
  return idx;
}

static int argmax_f32(const float* data, int n) {
  int idx = 0;
  float best = data[0];
  for (int i = 1; i < n; ++i) {
    if (data[i] > best) {
      best = data[i];
      idx = i;
    }
  }
  return idx;
}

extern "C" void app_main(void) {
  ESP_ERROR_CHECK(init_i2s_mic());
  if (!frontend_ready) init_frontend_tables();

  const tflite::Model* model = tflite::GetModel(kws_tflite_int8_kws_model_int8_DSCNN_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(kTag, "Model schema mismatch: model=%d runtime=%d", model->version(),
             TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddMaxPool2D();
  resolver.AddMean();

  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                              kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(kTag, "AllocateTensors() failed. Increase kTensorArenaSize.");
    return;
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  ESP_LOGI(kTag, "Ready. Capture=%dms, SR=%d, mel=%dx%d", kRecordMs, kSampleRateHz,
           kNMels, kNFrames);

  while (true) {
    if (!record_audio_1s(audio_pcm, kNumSamples)) {
      vTaskDelay(pdMS_TO_TICKS(500));
      continue;
    }

    pcm_to_mel_features(audio_pcm, mel_features);
    fill_model_input_from_mel(input, mel_features);

    if (interpreter.Invoke() != kTfLiteOk) {
      ESP_LOGE(kTag, "Invoke() failed.");
      vTaskDelay(pdMS_TO_TICKS(500));
      continue;
    }

    if (output->type == kTfLiteInt8) {
      int classes = output->bytes;
      int best = argmax_int8(output->data.int8, classes);
      ESP_LOGI(kTag, "Predicted class index: %d", best);
      for (int i = 0; i < classes; ++i) {
        float score = (output->data.int8[i] - output->params.zero_point) *
                      output->params.scale;
        ESP_LOGI(kTag, "  class %d -> q=%d score≈%.5f", i, output->data.int8[i], score);
      }
    } else if (output->type == kTfLiteFloat32) {
      int classes = output->bytes / static_cast<int>(sizeof(float));
      int best = argmax_f32(output->data.f, classes);
      ESP_LOGI(kTag, "Predicted class index: %d", best);
      for (int i = 0; i < classes; ++i) {
        ESP_LOGI(kTag, "  class %d -> %.5f", i, output->data.f[i]);
      }
    } else {
      ESP_LOGE(kTag, "Unsupported output type: %d", output->type);
    }

    vTaskDelay(pdMS_TO_TICKS(300));
  }
}