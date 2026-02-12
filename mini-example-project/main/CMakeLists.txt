
#
# Main component of TF Micro project 'micro_speech'.
#

idf_component_register(
    SRCS main.cc model.cc
    PRIV_REQUIRES spi_flash driver esp_timer test_data
    INCLUDE_DIRS "")

    # Reduce the level of paranoia to be able to compile sources
target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wno-maybe-uninitialized
    -Wno-missing-field-initializers
    -Wno-error=sign-compare
    -Wno-error=double-promotion
    -Wno-type-limits
    -DESP-NN)
