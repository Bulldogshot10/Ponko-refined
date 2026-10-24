// Ponko C ABI bridge for the Rust/UniFFI facade.
// The Kotlin-facing JNI surface is being replaced; the inference engine remains stable-diffusion.cpp.
#include "ponko_sd_c_api.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <android/log.h>
#include "ggml.h"
#include "gguf.h"
#ifdef PONKO_HAS_VULKAN
#include "ggml-vulkan.h"
#endif
#include "stable-diffusion.h"

#define PONKO_LOG_TAG "PonkoSdRust"
#define PONKO_LOGI(...) __android_log_print(ANDROID_LOG_INFO, PONKO_LOG_TAG, __VA_ARGS__)
#define PONKO_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, PONKO_LOG_TAG, __VA_ARGS__)

namespace {
constexpr size_t kSlotCount = 13;
enum SlotIndex {
    SLOT_MODEL = 0, SLOT_DIFFUSION, SLOT_HIGH_NOISE, SLOT_UNCOND, SLOT_CLIP_L, SLOT_CLIP_G,
    SLOT_CLIP_VISION, SLOT_T5XXL, SLOT_LLM, SLOT_LLM_VISION, SLOT_VAE, SLOT_AUDIO_VAE, SLOT_TAESD
};

struct PonkoHandle {
    sd_ctx_t* ctx = nullptr;
    std::atomic<bool> cancelled{false};
    bool vae_conv_direct = true;
    std::string backend_name = "auto";
};

struct CallbackState {
    ponko_step_callback_t step = nullptr;
    ponko_preview_callback_t preview = nullptr;
    void* user_data = nullptr;
};

std::mutex g_generation_mutex;
std::mutex g_last_params_mutex;
std::mutex g_vulkan_diagnostics_mutex;
std::string g_last_params;
std::string g_vulkan_diagnostics;
thread_local std::string g_last_error;

void setError(std::string message) {
    g_last_error = std::move(message);
}

size_t copyString(const std::string& source, char* out, size_t capacity) {
    if (out != nullptr && capacity > 0) {
        const size_t count = std::min(source.size(), capacity - 1);
        if (count > 0) std::memcpy(out, source.data(), count);
        out[count] = '\0';
    }
    return source.size();
}

bool stringIsNonempty(const char* text) {
    return text != nullptr && text[0] != '\0';
}

void ponkoLogCallback(enum ggml_log_level level, const char* text, void*) {
    if (text == nullptr || text[0] == '\0') return;
    std::fputs(text, stderr);
    std::fflush(stderr);
    const bool warning = level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR;
    const bool vulkan = std::strstr(text, "ggml_vulkan:") != nullptr ||
                        std::strstr(text, "Xclipse 940") != nullptr;
    if (warning || vulkan) {
        const int priority = level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR :
                             level == GGML_LOG_LEVEL_WARN ? ANDROID_LOG_WARN : ANDROID_LOG_INFO;
        __android_log_print(priority, vulkan ? "PonkoVulkan" : PONKO_LOG_TAG, "%s", text);
    }
    if (std::strstr(text, "Xclipse 940 profile detected") != nullptr ||
        std::strstr(text, "matrix cores:") != nullptr) {
        std::lock_guard<std::mutex> guard(g_vulkan_diagnostics_mutex);
        if (g_vulkan_diagnostics.find(text) == std::string::npos) {
            g_vulkan_diagnostics.append(text);
            if (g_vulkan_diagnostics.size() > 8192) {
                g_vulkan_diagnostics.erase(0, g_vulkan_diagnostics.size() - 8192);
            }
        }
    }
}

void progressCallback(int step, int steps, float, void* data) {
    auto* state = static_cast<CallbackState*>(data);
    if (state != nullptr && state->step != nullptr) state->step(state->user_data, step, steps);
}

void previewCallback(int step, int frame_count, sd_image_t* frames, bool, void* data) {
    auto* state = static_cast<CallbackState*>(data);
    if (state == nullptr || state->preview == nullptr || frames == nullptr || frame_count <= 0) return;
    const sd_image_t& image = frames[0];
    if (image.data == nullptr || image.width == 0 || image.height == 0 || image.channel == 0) return;
    const uint64_t pixel_count = static_cast<uint64_t>(image.width) * image.height;
    if (pixel_count > std::numeric_limits<size_t>::max() / 3) return;
    const size_t rgb_length = static_cast<size_t>(pixel_count) * 3;
    if (image.channel == 3) {
        state->preview(state->user_data, step, static_cast<int32_t>(image.width),
                       static_cast<int32_t>(image.height), image.data, rgb_length);
        return;
    }
    std::vector<uint8_t> rgb(rgb_length);
    for (size_t i = 0; i < static_cast<size_t>(pixel_count); ++i) {
        const uint8_t* src = image.data + i * image.channel;
        rgb[i * 3] = src[0];
        rgb[i * 3 + 1] = image.channel >= 3 ? src[1] : src[0];
        rgb[i * 3 + 2] = image.channel >= 3 ? src[2] : src[0];
    }
    state->preview(state->user_data, step, static_cast<int32_t>(image.width),
                   static_cast<int32_t>(image.height), rgb.data(), rgb.size());
}

bool validateQuantizedUnetFile(const std::string& path, ggml_type expected_type,
                               int64_t* unet_tensor_count, int64_t* quantized_tensor_count) {
    if (unet_tensor_count) *unet_tensor_count = 0;
    if (quantized_tensor_count) *quantized_tensor_count = 0;
    gguf_init_params params = { true, nullptr };
    gguf_context* context = gguf_init_from_file(path.c_str(), params);
    if (context == nullptr) {
        setError("quantized output is not a readable GGUF file");
        return false;
    }

    constexpr const char* prefix = "model.diffusion_model.";
    const size_t prefix_length = std::strlen(prefix);
    int64_t unet_count = 0;
    int64_t quantized_count = 0;
    int64_t outside_quant_count = 0;
    bool payload_complete = true;
    const uint64_t data_base = static_cast<uint64_t>(gguf_get_data_offset(context));
    uint64_t required_file_size = data_base;
    const uint64_t max_size = std::numeric_limits<uint64_t>::max();

    const int64_t tensor_count = gguf_get_n_tensors(context);
    for (int64_t i = 0; i < tensor_count; ++i) {
        const char* name = gguf_get_tensor_name(context, i);
        if (name == nullptr) continue;
        const bool is_unet = std::strncmp(name, prefix, prefix_length) == 0;
        const ggml_type type = gguf_get_tensor_type(context, i);
        if (is_unet) {
            ++unet_count;
            if (type == expected_type) ++quantized_count;
        } else if (type == expected_type) {
            ++outside_quant_count;
        }

        const uint64_t offset = static_cast<uint64_t>(gguf_get_tensor_offset(context, i));
        const uint64_t size = static_cast<uint64_t>(gguf_get_tensor_size(context, i));
        if (offset > max_size - data_base || size > max_size - data_base - offset) {
            payload_complete = false;
            break;
        }
        required_file_size = std::max(required_file_size, data_base + offset + size);
    }

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const std::streamoff actual_size = file ? static_cast<std::streamoff>(file.tellg()) : -1;
    payload_complete = payload_complete && actual_size >= 0 &&
        static_cast<uint64_t>(actual_size) >= required_file_size;
    gguf_free(context);

    if (unet_tensor_count) *unet_tensor_count = unet_count;
    if (quantized_tensor_count) *quantized_tensor_count = quantized_count;
    if (unet_count == 0 || quantized_count == 0 || outside_quant_count != 0 || !payload_complete) {
        setError("GGUF validation failed: UNet tensors, quantization scope, or payload size is invalid");
        return false;
    }
    return true;
}

bool parseQuantType(const std::string& name, sd_type_t* output) {
    if (name == "q4_0") *output = SD_TYPE_Q4_0;
    else if (name == "q5_0") *output = SD_TYPE_Q5_0;
    else if (name == "q8_0") *output = SD_TYPE_Q8_0;
    else {
        setError("supported UNet quantization types are q4_0, q5_0, and q8_0");
        return false;
    }
    return true;
}
} // namespace

extern "C" const char* ponko_sd_last_error(void) {
    return g_last_error.c_str();
}

extern "C" void* ponko_sd_create(const char* model_path, const char* vae_path, int32_t n_threads,
                                  int32_t wtype, int32_t flash_attn, int32_t vae_conv_direct,
                                  const char* backend, const char* const* slots, size_t slot_count) {
    g_last_error.clear();
    if (model_path == nullptr && (slots == nullptr || slot_count == 0)) {
        setError("model path or slot paths are required");
        return nullptr;
    }

    std::vector<std::string> slot_storage(kSlotCount);
    std::vector<const char*> slot_paths(kSlotCount, nullptr);
    if (slots != nullptr) {
        const size_t count = std::min(slot_count, kSlotCount);
        bool any_slot = false;
        for (size_t i = 0; i < count; ++i) {
            if (slots[i] != nullptr && slots[i][0] != '\0') {
                slot_storage[i] = slots[i];
                slot_paths[i] = slot_storage[i].c_str();
                any_slot = true;
            }
        }
        if (!any_slot && !stringIsNonempty(model_path)) {
            setError("at least one model slot path must be non-empty");
            return nullptr;
        }
    }

    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    const char* effective_model = slot_paths[SLOT_MODEL] != nullptr ? slot_paths[SLOT_MODEL] : model_path;
    const char* effective_vae = slot_paths[SLOT_VAE] != nullptr ? slot_paths[SLOT_VAE] : vae_path;
    params.model_path = effective_model;
    params.diffusion_model_path = slot_paths[SLOT_DIFFUSION];
    params.high_noise_diffusion_model_path = slot_paths[SLOT_HIGH_NOISE];
    params.uncond_diffusion_model_path = slot_paths[SLOT_UNCOND];
    params.clip_l_path = slot_paths[SLOT_CLIP_L];
    params.clip_g_path = slot_paths[SLOT_CLIP_G];
    params.clip_vision_path = slot_paths[SLOT_CLIP_VISION];
    params.t5xxl_path = slot_paths[SLOT_T5XXL];
    params.llm_path = slot_paths[SLOT_LLM];
    params.llm_vision_path = slot_paths[SLOT_LLM_VISION];
    params.vae_path = effective_vae;
    params.audio_vae_path = slot_paths[SLOT_AUDIO_VAE];
    params.taesd_path = slot_paths[SLOT_TAESD];
    params.n_threads = n_threads > 0 ? n_threads : sd_get_num_physical_cores();
    if (wtype >= 0) params.wtype = static_cast<sd_type_t>(wtype);
    params.enable_mmap = true;
    const bool flash = flash_attn != 0;
    params.flash_attn = flash;
    params.diffusion_flash_attn = flash;
    params.vae_conv_direct = vae_conv_direct != 0;
    params.lora_apply_mode = LORA_APPLY_AUTO;
    if (stringIsNonempty(backend)) params.backend = backend;

    char* dump = sd_ctx_params_to_str(&params);
    {
        std::lock_guard<std::mutex> guard(g_last_params_mutex);
        g_last_params = dump == nullptr ? "" : dump;
    }
    PONKO_LOGI("new_sd_ctx: %s", dump == nullptr ? "(null)" : dump);
    if (dump != nullptr) std::free(dump);

    sd_ctx_t* context = new_sd_ctx(&params);
    if (context == nullptr) {
        setError("stable-diffusion.cpp could not create a model context");
        return nullptr;
    }
    auto* handle = new (std::nothrow) PonkoHandle();
    if (handle == nullptr) {
        free_sd_ctx(context);
        setError("out of memory while allocating the native handle");
        return nullptr;
    }
    handle->ctx = context;
    handle->vae_conv_direct = params.vae_conv_direct;
    handle->backend_name = stringIsNonempty(backend) ? backend : "auto";
    PONKO_LOGI("context created; VAE direct-conv=%d backend=%s",
               handle->vae_conv_direct ? 1 : 0, handle->backend_name.c_str());
    return handle;
}

extern "C" int32_t ponko_sd_generate(void* opaque_handle, const ponko_generation_params_t* params,
                                      ponko_step_callback_t step_callback,
                                      ponko_preview_callback_t preview_callback, void* user_data,
                                      uint8_t** rgb_out, size_t* rgb_length_out,
                                      int32_t* width_out, int32_t* height_out) {
    g_last_error.clear();
    if (rgb_out) *rgb_out = nullptr;
    if (rgb_length_out) *rgb_length_out = 0;
    if (width_out) *width_out = 0;
    if (height_out) *height_out = 0;
    auto* handle = static_cast<PonkoHandle*>(opaque_handle);
    if (handle == nullptr || handle->ctx == nullptr || params == nullptr ||
        rgb_out == nullptr || rgb_length_out == nullptr || width_out == nullptr || height_out == nullptr) {
        setError("invalid handle, request, or output pointer");
        return 0;
    }
    if (params->width <= 0 || params->height <= 0 || params->steps <= 0) {
        setError("width, height and steps must be positive");
        return 0;
    }

    std::unique_lock<std::mutex> generation_guard(g_generation_mutex, std::try_to_lock);
    if (!generation_guard.owns_lock()) {
        setError("drawing engine is busy");
        return 0;
    }

    std::vector<sd_lora_t> loras;
    loras.reserve(params->lora_count);
    for (size_t i = 0; i < params->lora_count; ++i) {
        if (params->lora_paths == nullptr || params->lora_scales == nullptr ||
            params->lora_paths[i] == nullptr || params->lora_paths[i][0] == '\0') continue;
        sd_lora_t lora{};
        lora.is_high_noise = false;
        lora.multiplier = params->lora_scales[i];
        lora.path = params->lora_paths[i];
        loras.push_back(lora);
    }

    CallbackState callback_state{step_callback, preview_callback, user_data};
    handle->cancelled.store(false);
    sd_set_progress_callback(progressCallback, &callback_state);
    if (preview_callback != nullptr) sd_set_preview_callback(previewCallback, PREVIEW_PROJ, 1, true, false, &callback_state);
    else sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, true, false, nullptr);

    sd_img_gen_params_t generation;
    sd_img_gen_params_init(&generation);
    generation.prompt = params->prompt == nullptr ? "" : params->prompt;
    generation.negative_prompt = stringIsNonempty(params->negative_prompt) ? params->negative_prompt : nullptr;
    generation.width = params->width;
    generation.height = params->height;
    generation.seed = params->seed;
    generation.batch_count = 1;
    generation.strength = 1.0f;
    generation.clip_skip = std::max(1, std::min(12, params->clip_skip));
    generation.loras = loras.empty() ? nullptr : loras.data();
    generation.lora_count = static_cast<uint32_t>(loras.size());
    generation.sample_params.sample_steps = params->steps;
    generation.sample_params.eta = params->eta < 0.0f ? std::numeric_limits<float>::infinity() : params->eta;
    generation.sample_params.guidance.txt_cfg = params->cfg;
    generation.sample_params.sample_method = static_cast<sample_method_t>(params->sample_method);
    generation.sample_params.scheduler = static_cast<scheduler_t>(params->scheduler);
    generation.hires.enabled = params->hires_enabled != 0;
    generation.hires.upscaler = static_cast<sd_hires_upscaler_t>(params->hires_upscaler);
    generation.hires.model_path = stringIsNonempty(params->hires_model_path) ? params->hires_model_path : nullptr;
    generation.hires.scale = params->hires_scale;
    generation.hires.target_width = params->hires_target_width;
    generation.hires.target_height = params->hires_target_height;
    generation.hires.steps = params->hires_steps;
    generation.hires.denoising_strength = params->hires_denoising_strength;
    generation.hires.upscale_tile_size = params->hires_tile_size;

    if (params->init_image_rgb888 != nullptr && params->init_image_width > 0 && params->init_image_height > 0) {
        const uint64_t expected = static_cast<uint64_t>(params->init_image_width) *
                                  static_cast<uint64_t>(params->init_image_height) * 3ULL;
        if (expected != params->init_image_length) {
            sd_set_progress_callback(nullptr, nullptr);
            sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, true, false, nullptr);
            setError("img2img RGB888 buffer size does not match its dimensions");
            return 0;
        }
        generation.init_image.width = static_cast<uint32_t>(params->init_image_width);
        generation.init_image.height = static_cast<uint32_t>(params->init_image_height);
        generation.init_image.channel = 3;
        generation.init_image.data = const_cast<uint8_t*>(params->init_image_rgb888);
        float strength = params->strength;
        if (!(strength > 0.0f)) strength = 0.01f;
        if (strength >= 1.0f) strength = 0.99f;
        generation.strength = strength;
    }

    char* dump = sd_img_gen_params_to_str(&generation);
    PONKO_LOGI("generate_image: %s", dump == nullptr ? "(null)" : dump);
    if (dump != nullptr) std::free(dump);

    sd_image_t* images = nullptr;
    int image_count = 0;
    const auto start = std::chrono::steady_clock::now();
    bool ok = generate_image(handle->ctx, &generation, &images, &image_count);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    if (image_count > 0 && images != nullptr && images[0].data != nullptr) ok = true;
    PONKO_LOGI("BENCHMARK generate_image elapsed_ms=%lld mode=%s size=%dx%d steps=%d backend=%s vae_conv_direct=%d ok=%d",
               static_cast<long long>(elapsed_ms),
               generation.init_image.data == nullptr ? "txt2img" : "img2img",
               params->width, params->height, params->steps, handle->backend_name.c_str(),
               handle->vae_conv_direct ? 1 : 0, ok ? 1 : 0);

    int32_t success = 0;
    if (ok && image_count > 0 && images != nullptr && images[0].data != nullptr) {
        const sd_image_t& image = images[0];
        const uint64_t pixel_count = static_cast<uint64_t>(image.width) * image.height;
        if (pixel_count > std::numeric_limits<size_t>::max() / 3) {
            setError("generated image dimensions overflow the output buffer");
        } else {
            const size_t bytes = static_cast<size_t>(pixel_count) * 3;
            auto* rgb = static_cast<uint8_t*>(std::malloc(bytes));
            if (rgb == nullptr) {
                setError("out of memory allocating generated RGB888 pixels");
            } else {
                for (size_t i = 0; i < static_cast<size_t>(pixel_count); ++i) {
                    const uint8_t* src = image.data + i * image.channel;
                    rgb[i * 3] = src[0];
                    rgb[i * 3 + 1] = image.channel >= 3 ? src[1] : src[0];
                    rgb[i * 3 + 2] = image.channel >= 3 ? src[2] : src[0];
                }
                *rgb_out = rgb;
                *rgb_length_out = bytes;
                *width_out = static_cast<int32_t>(image.width);
                *height_out = static_cast<int32_t>(image.height);
                success = 1;
            }
        }
    } else {
        setError("stable-diffusion.cpp generation failed");
    }
    if (images != nullptr) free_sd_images(images, image_count);
    sd_set_progress_callback(nullptr, nullptr);
    sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, true, false, nullptr);
    return success;
}

extern "C" int32_t ponko_sd_cancel(void* opaque_handle) {
    auto* handle = static_cast<PonkoHandle*>(opaque_handle);
    if (handle == nullptr || handle->ctx == nullptr) {
        setError("cannot cancel a null native handle");
        return 0;
    }
    handle->cancelled.store(true);
    sd_cancel_generation(handle->ctx, SD_CANCEL_ALL);
    return 1;
}

extern "C" void ponko_sd_free(void* opaque_handle) {
    auto* handle = static_cast<PonkoHandle*>(opaque_handle);
    if (handle == nullptr) return;
    if (handle->ctx != nullptr) free_sd_ctx(handle->ctx);
    delete handle;
}

extern "C" void ponko_sd_free_buffer(void* buffer) {
    std::free(buffer);
}

extern "C" size_t ponko_sd_info(char* out, size_t capacity) {
    const std::string info = std::string("sd.cpp ") + sd_version() + " | " + sd_get_system_info();
    return copyString(info, out, capacity);
}

extern "C" size_t ponko_sd_vulkan_diagnostics(char* out, size_t capacity) {
    std::string diagnostics;
#ifdef PONKO_HAS_VULKAN
    char device_diagnostics[2048] = {};
    const int device_count = ggml_backend_vk_get_device_count();
    if (device_count > 0) ggml_backend_vk_get_device_diagnostics(0, device_diagnostics, sizeof(device_diagnostics));
    else std::snprintf(device_diagnostics, sizeof(device_diagnostics), "No Vulkan devices enumerated by ggml-vulkan");
    diagnostics = device_diagnostics;
#else
    diagnostics = "Vulkan backend was not compiled into this build";
#endif
    {
        std::lock_guard<std::mutex> guard(g_vulkan_diagnostics_mutex);
        if (!g_vulkan_diagnostics.empty() && diagnostics.find(g_vulkan_diagnostics) == std::string::npos) {
            if (!diagnostics.empty()) diagnostics += "\n";
            diagnostics += g_vulkan_diagnostics;
        }
    }
    return copyString(diagnostics, out, capacity);
}

extern "C" size_t ponko_sd_last_params(char* out, size_t capacity) {
    std::lock_guard<std::mutex> guard(g_last_params_mutex);
    return copyString(g_last_params, out, capacity);
}

extern "C" size_t ponko_sd_list_devices(char* out, size_t capacity) {
    return sd_list_devices(out, capacity);
}

extern "C" int32_t ponko_sd_quantize_unet(const char* input_path, const char* output_path,
                                           const char* quant_type, int32_t diffusion_slot, int32_t n_threads) {
    g_last_error.clear();
    if (!stringIsNonempty(input_path) || !stringIsNonempty(output_path) || !stringIsNonempty(quant_type)) {
        setError("quantization input, output and type must be provided");
        return 0;
    }
    sd_type_t quant_type_enum;
    if (!parseQuantType(quant_type, &quant_type_enum)) return 0;
    std::ifstream source(input_path, std::ios::binary);
    if (!source.good()) {
        setError("quantization source file does not exist");
        return 0;
    }
    std::ifstream target(output_path, std::ios::binary);
    if (target.good()) {
        setError("refusing to overwrite an existing output file");
        return 0;
    }
    const std::string temporary = std::string(output_path) + ".partial.gguf";
    std::remove(temporary.c_str());
    const std::string rules = std::string("^model\\.diffusion_model\\..*=") + quant_type;
    const char* model = diffusion_slot != 0 ? nullptr : input_path;
    const char* diffusion = diffusion_slot != 0 ? input_path : nullptr;
    const int threads = std::max(1, std::min(6, n_threads));
    const bool converted = convert_with_components(model, nullptr, nullptr, nullptr, diffusion, nullptr,
                                                    temporary.c_str(), SD_TYPE_F16, rules.c_str(), true, threads);
    if (!converted) {
        std::remove(temporary.c_str());
        setError("stable-diffusion.cpp quantization conversion failed");
        return 0;
    }
    int64_t unet_count = 0;
    int64_t quantized_count = 0;
    if (!validateQuantizedUnetFile(temporary, static_cast<ggml_type>(quant_type_enum),
                                   &unet_count, &quantized_count)) {
        std::remove(temporary.c_str());
        return 0;
    }
    if (std::rename(temporary.c_str(), output_path) != 0) {
        std::remove(temporary.c_str());
        setError("failed to atomically publish quantized GGUF output");
        return 0;
    }
    PONKO_LOGI("UNet quantization complete: tensors=%lld selected_type=%lld output=%s",
               static_cast<long long>(unet_count), static_cast<long long>(quantized_count), output_path);
    return 1;
}

extern "C" int32_t ponko_sd_apply_resadapter(const char* input_model_path, const char* vae_path,
                                              const char* normalization_path, const char* output_path,
                                              const char* quant_type, int32_t n_threads) {
    g_last_error.clear();
    if (!stringIsNonempty(input_model_path) || !stringIsNonempty(normalization_path) ||
        !stringIsNonempty(output_path) || !stringIsNonempty(quant_type)) {
        setError("ResAdapter base model, normalization, output and quantization type are required");
        return 0;
    }
    sd_type_t quant_type_enum;
    if (!parseQuantType(quant_type, &quant_type_enum)) return 0;
    std::ifstream source(input_model_path, std::ios::binary);
    std::ifstream normalization(normalization_path, std::ios::binary);
    std::ifstream target(output_path, std::ios::binary);
    if (!source.good() || !normalization.good()) {
        setError("ResAdapter base model or normalization file does not exist");
        return 0;
    }
    if (target.good()) {
        setError("refusing to overwrite an existing ResAdapter output file");
        return 0;
    }
    const std::string temporary = std::string(output_path) + ".partial.gguf";
    std::remove(temporary.c_str());
    const std::string rules = std::string("^model\\.diffusion_model\\..*=") + quant_type;
    const int threads = std::max(1, std::min(6, n_threads));
    const bool converted = convert_with_components(input_model_path, nullptr, nullptr, nullptr,
        normalization_path, stringIsNonempty(vae_path) ? vae_path : nullptr, temporary.c_str(),
        SD_TYPE_F16, rules.c_str(), true, threads);
    if (!converted) {
        std::remove(temporary.c_str());
        setError("ResAdapter normalization merge/conversion failed");
        return 0;
    }
    int64_t unet_count = 0;
    int64_t quantized_count = 0;
    if (!validateQuantizedUnetFile(temporary, static_cast<ggml_type>(quant_type_enum),
                                   &unet_count, &quantized_count)) {
        std::remove(temporary.c_str());
        return 0;
    }
    if (std::rename(temporary.c_str(), output_path) != 0) {
        std::remove(temporary.c_str());
        setError("failed to atomically publish ResAdapter output");
        return 0;
    }
    PONKO_LOGI("ResAdapter complete: tensors=%lld selected_type=%lld output=%s",
               static_cast<long long>(unet_count), static_cast<long long>(quantized_count), output_path);
    return 1;
}
