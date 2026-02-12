// ESP-IDF + INMP441 + TensorFlow Lite Micro example
// Records audio via I2S, computes MelSpectrogram-like features, then runs inference.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <new>

#include "driver/i2s.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "model.h"  // generated model byte array

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr const char* kTag = "kws_inmp441";
// Configure this list to match your model's output class order exactly.
// Example:
//   {"fernando", "noise", "yes", "no", "unknown"}
constexpr const char* kLabels[] = {
    "fernando",
    "gross",
    "kaffee",
    "klein",
    "noise",
};
constexpr int kNumLabels = sizeof(kLabels) / sizeof(kLabels[0]);
constexpr const char* kNoiseLabel = "noise";

// ---- Audio capture config (INMP441) ----
constexpr i2s_port_t kI2sPort = I2S_NUM_0;
constexpr int kSampleRateHz = 16000;
constexpr int kRecordMs = 1000;  // 1 second window
constexpr int kNumSamples = (kSampleRateHz * kRecordMs) / 1000;

// Set these pins to your wiring.
constexpr gpio_num_t kPinBck = GPIO_NUM_26;   // SCK/BCLK
constexpr gpio_num_t kPinWs = GPIO_NUM_25;    // WS/LRCLK
constexpr gpio_num_t kPinData = GPIO_NUM_33;  // SD

// ---- Feature config (aligned to training scripts) ----
constexpr int kNfft = 1024;
constexpr int kHop = 512;
constexpr int kNMels = 64;
constexpr int kNFrames = 32;
constexpr int kFftBins = (kNfft / 2) + 1;
constexpr int kWdtYieldEveryBins = 4;
constexpr int kWdtYieldEverySamples = 128;

// ---- TFLM config ----
constexpr size_t kTensorArenaSizeTarget = 96 * 1024;
constexpr size_t kTensorArenaMin = 32 * 1024;
constexpr size_t kTensorArenaStep = 4 * 1024;
uint8_t* tensor_arena = nullptr;
size_t tensor_arena_size = 0;

// ---- Runtime buffers (heap allocated to avoid .dram0.bss overflow) ----
int16_t* audio_pcm = nullptr;    // kNumSamples
float* mel_features = nullptr;   // kNMels * kNFrames
float* hann_window = nullptr;    // kNfft
int* mel_bin_points = nullptr;   // kNMels + 2
float* stft_frame = nullptr;     // kNfft
float* stft_power = nullptr;     // kFftBins
bool frontend_ready = false;
}  // namespace

static float hz_to_mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static float mel_to_hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

static void* alloc_heap_pref_psram(size_t bytes) {
  void* p = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (p != nullptr) return p;
  return heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
}

static bool alloc_frontend_buffers() {
  audio_pcm = static_cast<int16_t*>(alloc_heap_pref_psram(sizeof(int16_t) * kNumSamples));
  mel_features = static_cast<float*>(alloc_heap_pref_psram(sizeof(float) * kNMels * kNFrames));
  hann_window = static_cast<float*>(alloc_heap_pref_psram(sizeof(float) * kNfft));
  mel_bin_points = static_cast<int*>(alloc_heap_pref_psram(sizeof(int) * (kNMels + 2)));
  stft_frame = static_cast<float*>(alloc_heap_pref_psram(sizeof(float) * kNfft));
  stft_power = static_cast<float*>(alloc_heap_pref_psram(sizeof(float) * kFftBins));

  if (audio_pcm == nullptr || mel_features == nullptr || hann_window == nullptr ||
      mel_bin_points == nullptr || stft_frame == nullptr || stft_power == nullptr) {
    ESP_LOGE(kTag, "Frontend buffer allocation failed.");
    return false;
  }

  ESP_LOGI(kTag,
           "Frontend buffers allocated (audio=%uB mel=%uB window=%uB bins=%uB frame=%uB power=%uB).",
           static_cast<unsigned>(sizeof(int16_t) * kNumSamples),
           static_cast<unsigned>(sizeof(float) * kNMels * kNFrames),
           static_cast<unsigned>(sizeof(float) * kNfft),
           static_cast<unsigned>(sizeof(int) * (kNMels + 2)),
           static_cast<unsigned>(sizeof(float) * kNfft),
           static_cast<unsigned>(sizeof(float) * kFftBins));
  return true;
}

static void init_frontend_tables() {
  for (int n = 0; n < kNfft; ++n) {
    hann_window[n] = 0.5f - 0.5f * cosf((2.0f * static_cast<float>(M_PI) * n) / (kNfft - 1));
  }

  const float mel_min = hz_to_mel(0.0f);
  const float mel_max = hz_to_mel(static_cast<float>(kSampleRateHz) / 2.0f);
  for (int i = 0; i < kNMels + 2; ++i) {
    float mel = mel_min + (mel_max - mel_min) * (static_cast<float>(i) / (kNMels + 1));
    float hz = mel_to_hz(mel);
    int bin = static_cast<int>(floorf((kNfft + 1) * hz / kSampleRateHz));
    if (bin < 0) bin = 0;
    if (bin > kFftBins - 1) bin = kFftBins - 1;
    mel_bin_points[i] = bin;
  }

  frontend_ready = true;
}

static esp_err_t init_i2s_mic() {
  // Uses legacy I2S driver for compatibility with older examples.
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
  int32_t i2s_raw[256];

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


static inline void cooperative_wdt_yield() {
  // Allow IDLE task to run so task watchdog does not fire during heavy DSP loops.
  vTaskDelay(pdMS_TO_TICKS(1));
}

static void pcm_to_mel_features(const int16_t* pcm, float* out_mel) {
  float max_abs = 1e-9f;
  for (int i = 0; i < kNumSamples; ++i) {
    float x = static_cast<float>(pcm[i]) / 32768.0f;
    float a = fabsf(x);
    if (a > max_abs) max_abs = a;
  }

  for (int t = 0; t < kNFrames; ++t) {
    const int frame_center = t * kHop;

    for (int n = 0; n < kNfft; ++n) {
      int src = frame_center + n - (kNfft / 2);
      float x = 0.0f;
      if (src >= 0 && src < kNumSamples) {
        x = (static_cast<float>(pcm[src]) / 32768.0f) / max_abs;
      }
      stft_frame[n] = x * hann_window[n];
    }

    for (int k = 0; k < kFftBins; ++k) {
      if ((k % kWdtYieldEveryBins) == 0) cooperative_wdt_yield();
      float re = 0.0f;
      float im = 0.0f;

      // Iterative twiddle update: avoids costly cosf/sinf in the inner sample loop.
      const float delta = -2.0f * static_cast<float>(M_PI) * k / kNfft;
      const float c = cosf(delta);
      const float si = sinf(delta);
      float wr = 1.0f;
      float wi = 0.0f;

      for (int n = 0; n < kNfft; ++n) {
        if ((n % kWdtYieldEverySamples) == 0) cooperative_wdt_yield();
        const float x = stft_frame[n];
        re += x * wr;
        im += x * wi;

        const float next_wr = (wr * c) - (wi * si);
        wi = (wr * si) + (wi * c);
        wr = next_wr;
      }
      stft_power[k] = (re * re) + (im * im);
    }

    for (int m = 0; m < kNMels; ++m) {
      int left = mel_bin_points[m];
      int center = mel_bin_points[m + 1];
      int right = mel_bin_points[m + 2];
      if (center <= left) center = left + 1;
      if (right <= center) right = center + 1;
      if (right > kFftBins - 1) right = kFftBins - 1;

      float e = 0.0f;
      for (int k = left; k < center && k < kFftBins; ++k) {
        float w = (k - left) / static_cast<float>(center - left);
        e += w * stft_power[k];
      }
      for (int k = center; k <= right && k < kFftBins; ++k) {
        float w = (right - k) / static_cast<float>(right - center);
        e += w * stft_power[k];
      }
      out_mel[m * kNFrames + t] = e;
    }

    cooperative_wdt_yield();
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

static void log_heap_state(const char* phase) {
  size_t free8 = heap_caps_get_free_size(MALLOC_CAP_8BIT);
  size_t largest8 = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
  ESP_LOGI(kTag, "%s heap: free_8bit=%u largest_8bit_block=%u", phase,
           static_cast<unsigned>(free8), static_cast<unsigned>(largest8));
}

static uint8_t* alloc_tensor_arena_exact(size_t sz) {
  uint8_t* p = static_cast<uint8_t*>(
      heap_caps_malloc(sz, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (p != nullptr) {
    ESP_LOGI(kTag, "Tensor arena candidate in PSRAM: %u bytes", static_cast<unsigned>(sz));
    return p;
  }

  p = static_cast<uint8_t*>(
      heap_caps_malloc(sz, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (p != nullptr) {
    ESP_LOGI(kTag, "Tensor arena candidate in internal RAM: %u bytes", static_cast<unsigned>(sz));
  }
  return p;
}


static bool is_noise_label(int idx) {
  if (idx < 0 || idx >= kNumLabels) return false;
  return strcmp(kLabels[idx], kNoiseLabel) == 0;
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

static void kws_inference_task(void* arg) {
  (void)arg;
  ESP_LOGI(kTag, "kws task started on core %d", xPortGetCoreID());

  ESP_ERROR_CHECK(init_i2s_mic());

  log_heap_state("boot");

  if (!alloc_frontend_buffers()) {
    ESP_LOGE(kTag, "Not enough heap/PSRAM for frontend buffers.");
    ESP_LOGE(kTag, "Try reducing arena target or using optimized frontend (no naive DFT).");
    vTaskDelete(nullptr);
    return;
  }
  if (!frontend_ready) init_frontend_tables();
  const tflite::Model* tfl_model = tflite::GetModel(kws_tflite_int8_kws_model_int8_DSCNN_tflite);

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

  tflite::MicroInterpreter* interpreter = nullptr;
  for (size_t sz = kTensorArenaSizeTarget; sz >= kTensorArenaMin; sz -= kTensorArenaStep) {
    tensor_arena = alloc_tensor_arena_exact(sz);
    if (tensor_arena == nullptr) {
      if (sz == kTensorArenaMin) break;
      continue;
    }

    interpreter = new (std::nothrow)
        tflite::MicroInterpreter(tfl_model, resolver, tensor_arena, sz);
    if (interpreter == nullptr) {
      heap_caps_free(tensor_arena);
      tensor_arena = nullptr;
      if (sz == kTensorArenaMin) break;
      continue;
    }

    if (interpreter->AllocateTensors() == kTfLiteOk) {
      tensor_arena_size = sz;
      ESP_LOGI(kTag, "AllocateTensors() OK with arena=%u bytes", static_cast<unsigned>(sz));
      break;
    }

    ESP_LOGW(kTag, "AllocateTensors() failed with arena=%u bytes", static_cast<unsigned>(sz));
    delete interpreter;
    interpreter = nullptr;
    heap_caps_free(tensor_arena);
    tensor_arena = nullptr;

    if (sz == kTensorArenaMin) break;
  }

  if (interpreter == nullptr) {
    log_heap_state("arena_fail");
    ESP_LOGE(kTag, "Could not initialize interpreter between %u KB and %u KB arena.",
             static_cast<unsigned>(kTensorArenaMin / 1024),
             static_cast<unsigned>(kTensorArenaSizeTarget / 1024));
    vTaskDelete(nullptr);
    return;
  }

  TfLiteTensor* input = interpreter->input(0);
  TfLiteTensor* output = interpreter->output(0);

  int output_classes = 0;
  if (output->type == kTfLiteInt8) {
    output_classes = output->bytes;
  } else if (output->type == kTfLiteFloat32) {
    output_classes = output->bytes / static_cast<int>(sizeof(float));
  }

  if (output_classes > kNumLabels) {
    ESP_LOGW(kTag,
             "Model outputs %d classes, but only %d labels are configured. "
             "Update kLabels[] to match model order.",
             output_classes, kNumLabels);
  }

  ESP_LOGI(kTag, "Ready. Capture=%dms, SR=%d, mel=%dx%d", kRecordMs, kSampleRateHz,
           kNMels, kNFrames);
  ESP_LOGI(kTag, "stack high-water mark: %u words",
           static_cast<unsigned>(uxTaskGetStackHighWaterMark(nullptr)));

  while (true) {
    if (!record_audio_1s(audio_pcm, kNumSamples)) {
      vTaskDelay(pdMS_TO_TICKS(500));
      continue;
    }

    pcm_to_mel_features(audio_pcm, mel_features);
    fill_model_input_from_mel(input, mel_features);

    int64_t t0_us = esp_timer_get_time();
    if (interpreter->Invoke() != kTfLiteOk) {
      ESP_LOGE(kTag, "Invoke() failed.");
      vTaskDelay(pdMS_TO_TICKS(500));
      continue;
    }
    int64_t t1_us = esp_timer_get_time();
    ESP_LOGI(kTag, "Inference time: %.2f ms", (t1_us - t0_us) / 1000.0f);

    if (output->type == kTfLiteInt8) {
      int classes = output->bytes;
      int best = argmax_int8(output->data.int8, classes);
      const char* best_label = (best < kNumLabels) ? kLabels[best] : "unknown";
      if (is_noise_label(best)) {
        // Suppress logs when dominant class is noise.
      } else {
        ESP_LOGI(kTag, "Predicted class index: %d (%s)", best, best_label);
        for (int i = 0; i < classes; ++i) {
          float score = (output->data.int8[i] - output->params.zero_point) *
                        output->params.scale;
          const char* label = (i < kNumLabels) ? kLabels[i] : "unknown";
          ESP_LOGI(kTag, "  class %d (%s) -> q=%d score≈%.5f", i, label, output->data.int8[i], score);
        }
      }
    } else if (output->type == kTfLiteFloat32) {
      int classes = output->bytes / static_cast<int>(sizeof(float));
      int best = argmax_f32(output->data.f, classes);
      const char* best_label = (best < kNumLabels) ? kLabels[best] : "unknown";
      if (is_noise_label(best)) {
        // Suppress logs when dominant class is noise.
      } else {
        ESP_LOGI(kTag, "Predicted class index: %d (%s)", best, best_label);
        for (int i = 0; i < classes; ++i) {
          const char* label = (i < kNumLabels) ? kLabels[i] : "unknown";
          ESP_LOGI(kTag, "  class %d (%s) -> %.5f", i, label, output->data.f[i]);
        }
      }
    } else {
      ESP_LOGE(kTag, "Unsupported output type: %d", output->type);
    }

    vTaskDelay(pdMS_TO_TICKS(300));
  }
}

extern "C" void app_main(void) {
  constexpr uint32_t kTaskStackBytes = 16384;
  constexpr UBaseType_t kTaskPriority = 2;

  BaseType_t ok = xTaskCreatePinnedToCore(kws_inference_task, "kws_task", kTaskStackBytes,
                                           nullptr, kTaskPriority, nullptr, 1);
  if (ok != pdPASS) {
    ESP_LOGE(kTag, "Failed to create kws task.");
    return;
  }

  ESP_LOGI(kTag, "Spawned kws task on core 1");
}