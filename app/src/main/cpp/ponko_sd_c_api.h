#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ponko_step_callback_t)(void* user_data, int32_t step, int32_t steps);
typedef void (*ponko_preview_callback_t)(void* user_data, int32_t step, int32_t width,
                                         int32_t height, const uint8_t* rgb, size_t length);

typedef struct ponko_generation_params {
    const char* prompt;
    const char* negative_prompt;
    const char* const* lora_paths;
    const float* lora_scales;
    size_t lora_count;
    int32_t width;
    int32_t height;
    int32_t steps;
    float cfg;
    int64_t seed;
    int32_t sample_method;
    int32_t scheduler;
    float eta;
    int32_t hires_enabled;
    int32_t hires_upscaler;
    const char* hires_model_path;
    float hires_scale;
    int32_t hires_target_width;
    int32_t hires_target_height;
    int32_t hires_steps;
    float hires_denoising_strength;
    int32_t hires_tile_size;
    int32_t clip_skip;
    const uint8_t* init_image_rgb888;
    size_t init_image_length;
    int32_t init_image_width;
    int32_t init_image_height;
    float strength;
    int32_t adetailer_enabled;
    const char* adetailer_detector_path;
    const char* adetailer_prompt;
    const char* adetailer_negative_prompt;
    const char* adetailer_extra_args;
} ponko_generation_params_t;

const char* ponko_sd_last_error(void);
void* ponko_sd_create(const char* model_path, const char* vae_path, int32_t n_threads,
                      int32_t wtype, int32_t flash_attn, int32_t vae_conv_direct,
                      const char* backend, const char* const* slots, size_t slot_count);
int32_t ponko_sd_generate(void* handle, const ponko_generation_params_t* params,
                          ponko_step_callback_t step_callback,
                          ponko_preview_callback_t preview_callback, void* user_data,
                          uint8_t** rgb_out, size_t* rgb_length_out,
                          int32_t* width_out, int32_t* height_out);
int32_t ponko_sd_cancel(void* handle);
void ponko_sd_free(void* handle);
void ponko_sd_free_buffer(void* buffer);
size_t ponko_sd_info(char* out, size_t capacity);
size_t ponko_sd_vulkan_diagnostics(char* out, size_t capacity);
size_t ponko_sd_last_params(char* out, size_t capacity);
size_t ponko_sd_list_devices(char* out, size_t capacity);
int32_t ponko_sd_fuse_loras(const char* input_model_path,
                              const char* const* lora_paths, const float* lora_scales,
                              size_t lora_count, const char* output_path,
                              const char* quant_type, int32_t n_threads);
int32_t ponko_sd_quantize_unet(const char* input_path, const char* output_path,
                               const char* quant_type, int32_t diffusion_slot, int32_t n_threads);
int32_t ponko_sd_apply_resadapter(const char* input_model_path, const char* vae_path,
                                  const char* normalization_path, const char* output_path,
                                  const char* quant_type, int32_t n_threads);

#ifdef __cplusplus
}
#endif
