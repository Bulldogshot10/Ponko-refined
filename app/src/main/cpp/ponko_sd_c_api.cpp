// Ponko C ABI bridge for the Rust/UniFFI facade.
// The Kotlin-facing JNI surface is being replaced; the inference engine remains stable-diffusion.cpp.
#include "ponko_sd_c_api.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
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
#include "sd/src/model_io/safetensors_io.h"

#define PONKO_LOG_TAG "PonkoSdRust"
#define PONKO_LOGI(...) __android_log_print(ANDROID_LOG_INFO, PONKO_LOG_TAG, __VA_ARGS__)
#define PONKO_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, PONKO_LOG_TAG, __VA_ARGS__)

namespace {
constexpr size_t kSlotCount = 16;
enum SlotIndex {
    SLOT_MODEL = 0, SLOT_DIFFUSION, SLOT_HIGH_NOISE, SLOT_UNCOND, SLOT_CLIP_L, SLOT_CLIP_G,
    SLOT_CLIP_VISION, SLOT_T5XXL, SLOT_LLM, SLOT_LLM_VISION, SLOT_VAE, SLOT_AUDIO_VAE,
    SLOT_TAESD, SLOT_CONTROL_NET, SLOT_IP_ADAPTER, SLOT_EMBEDDING
};

struct PonkoHandle {
    sd_ctx_t* ctx = nullptr;
    std::atomic<bool> cancelled{false};
    bool vae_conv_direct = true;
    int n_threads = 4;
    std::string backend_name = "auto";
    adetailer_ctx_t* adetailer = nullptr;
    std::string adetailer_detector_path;
    bool has_control_net = false;
    bool has_ip_adapter = false;
};

struct CallbackState {
    ponko_step_callback_t step = nullptr;
    ponko_preview_callback_t preview = nullptr;
    void* user_data = nullptr;
};

std::mutex g_generation_mutex;
std::once_flag g_log_init;
std::mutex g_last_params_mutex;
std::mutex g_vulkan_diagnostics_mutex;
std::string g_last_params;
std::string g_vulkan_diagnostics;
thread_local std::string g_last_error;
thread_local std::string g_last_native_error_log;

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

void captureNativeErrorLog(const char* text) {
    constexpr size_t kMaxDiagnosticBytes = 8192;
    if (text == nullptr || text[0] == '\0' ||
        g_last_native_error_log.size() >= kMaxDiagnosticBytes ||
        g_last_native_error_log.find(text) != std::string::npos) {
        return;
    }
    if (!g_last_native_error_log.empty() && g_last_native_error_log.back() != '\n' &&
        g_last_native_error_log.size() < kMaxDiagnosticBytes) {
        g_last_native_error_log.push_back('\n');
    }
    const size_t remaining = kMaxDiagnosticBytes - g_last_native_error_log.size();
    g_last_native_error_log.append(text, std::min(std::strlen(text), remaining));
}

void ponkoLogCallback(enum ggml_log_level level, const char* text, void*) {
    if (text == nullptr || text[0] == '\0') return;
    if (level == GGML_LOG_LEVEL_ERROR) captureNativeErrorLog(text);
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

// Forward stable-diffusion.cpp's LOG_INFO/LOG_WARN stream to Logcat as well as stderr.
// Without this callback, fusion-stage and RSS diagnostics are silently dropped.
void ponkoSdLogCallback(enum sd_log_level_t level, const char* text, void*) {
    if (text == nullptr || text[0] == '\0') return;
    if (level == SD_LOG_ERROR) captureNativeErrorLog(text);
    std::fputs(text, stderr);
    std::fflush(stderr);
    const int priority = level == SD_LOG_ERROR ? ANDROID_LOG_ERROR :
                         level == SD_LOG_WARN ? ANDROID_LOG_WARN :
                         level == SD_LOG_DEBUG ? ANDROID_LOG_DEBUG :
                         level == SD_LOG_VERBOSE ? ANDROID_LOG_VERBOSE : ANDROID_LOG_INFO;
    __android_log_print(priority, PONKO_LOG_TAG, "%s", text);
}

void initializePonkoLogging() {
    std::call_once(g_log_init, [] {
        ggml_log_set(ponkoLogCallback, nullptr);
        sd_set_log_callback(ponkoSdLogCallback, nullptr);
    });
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

bool validateQuantizedUnetFile(const std::string& path, enum ggml_type expected_type,
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
        const enum ggml_type type = gguf_get_tensor_type(context, i);
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

bool parseQuantType(const std::string& name, enum sd_type_t* output) {
    if (name == "q4_0") *output = SD_TYPE_Q4_0;
    else if (name == "q5_0") *output = SD_TYPE_Q5_0;
    else if (name == "q8_0") *output = SD_TYPE_Q8_0;
    else {
        setError("supported UNet quantization types are q4_0, q5_0, and q8_0");
        return false;
    }
    return true;
}

bool validateReadableGguf(const std::string& path) {
    gguf_init_params params = { true, nullptr };
    gguf_context* context = gguf_init_from_file(path.c_str(), params);
    if (context == nullptr) {
        setError("LoRA fusion output is not a readable GGUF file");
        return false;
    }
    const int64_t tensor_count = gguf_get_n_tensors(context);
    const uint64_t data_base = static_cast<uint64_t>(gguf_get_data_offset(context));
    uint64_t required_file_size = data_base;
    const uint64_t max_size = std::numeric_limits<uint64_t>::max();
    bool valid = tensor_count > 0;
    for (int64_t i = 0; valid && i < tensor_count; ++i) {
        const uint64_t offset = static_cast<uint64_t>(gguf_get_tensor_offset(context, i));
        const uint64_t size = static_cast<uint64_t>(gguf_get_tensor_size(context, i));
        if (offset > max_size - data_base || size > max_size - data_base - offset) {
            valid = false;
            break;
        }
        required_file_size = std::max(required_file_size, data_base + offset + size);
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const std::streamoff actual_size = file ? static_cast<std::streamoff>(file.tellg()) : -1;
    valid = valid && actual_size >= 0 && static_cast<uint64_t>(actual_size) >= required_file_size;
    gguf_free(context);
    if (!valid) setError("LoRA fusion GGUF validation failed: tensor metadata or payload is incomplete");
    return valid;
}
} // namespace

extern "C" const char* ponko_sd_last_error(void) {
    return g_last_error.c_str();
}

extern "C" void* ponko_sd_create(const char* model_path, const char* vae_path, int32_t n_threads,
                                  int32_t wtype, int32_t flash_attn, int32_t vae_conv_direct,
                                  const char* backend, const char* const* slots, size_t slot_count) {
    g_last_error.clear();
    initializePonkoLogging();
    g_last_native_error_log.clear();
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
        if (!any_slot) {
            setError("slots were supplied but every model slot path is empty");
            return nullptr;
        }
    }

    std::string embedding_name_storage;
    sd_embedding_t embedding{};
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    const char* effective_model = slot_paths[SLOT_MODEL] != nullptr
        ? slot_paths[SLOT_MODEL] : (stringIsNonempty(model_path) ? model_path : nullptr);
    const char* effective_vae = slot_paths[SLOT_VAE] != nullptr
        ? slot_paths[SLOT_VAE] : (stringIsNonempty(vae_path) ? vae_path : nullptr);
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
    params.control_net_path = slot_paths[SLOT_CONTROL_NET];
    params.ip_adapter_path = slot_paths[SLOT_IP_ADAPTER];
    if (slot_paths[SLOT_EMBEDDING] != nullptr) {
        const char* path = slot_paths[SLOT_EMBEDDING];
        const char* filename = std::strrchr(path, '/');
        filename = filename == nullptr ? path : filename + 1;
        embedding_name_storage = filename;
        const size_t extension = embedding_name_storage.find_last_of('.');
        if (extension != std::string::npos && extension > 0) embedding_name_storage.resize(extension);
        if (embedding_name_storage.empty()) embedding_name_storage = "embedding";
        embedding.name = embedding_name_storage.c_str();
        embedding.path = path;
        params.embeddings = &embedding;
        params.embedding_count = 1;
    }
    params.n_threads = n_threads > 0 ? n_threads : sd_get_num_physical_cores();
    if (wtype >= 0) params.wtype = static_cast<enum sd_type_t>(wtype);
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
        std::string detail = g_last_native_error_log;
        while (!detail.empty() && (detail.back() == '\n' || detail.back() == '\r')) {
            detail.pop_back();
        }
        if (detail.empty()) {
            setError("stable-diffusion.cpp could not create a model context");
        } else {
            setError("stable-diffusion.cpp could not create a model context: " + detail);
        }
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
    handle->has_control_net = slot_paths[SLOT_CONTROL_NET] != nullptr;
    handle->has_ip_adapter = slot_paths[SLOT_IP_ADAPTER] != nullptr;
    handle->n_threads = params.n_threads;
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
    auto validateOptionalRgb888 = [&](const uint8_t* pixels, size_t length,
                                      int32_t width, int32_t height, const char* label) {
        if (pixels == nullptr) {
            if (length != 0 || width != 0 || height != 0) {
                setError(std::string(label) + " image dimensions/length were set without a pixel buffer");
                return false;
            }
            return true;
        }
        if (width <= 0 || height <= 0) {
            setError(std::string(label) + " image dimensions must be positive");
            return false;
        }
        const uint64_t pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
        if (pixel_count > std::numeric_limits<size_t>::max() / 3) {
            setError(std::string(label) + " image size overflows the RGB888 buffer");
            return false;
        }
        const size_t expected = static_cast<size_t>(pixel_count) * 3;
        if (length != expected) {
            setError(std::string(label) + " image length does not match width * height * 3 RGB888 bytes");
            return false;
        }
        return true;
    };
    if (!validateOptionalRgb888(params->control_image_rgb888, params->control_image_length,
                                params->control_image_width, params->control_image_height, "ControlNet") ||
        !validateOptionalRgb888(params->ip_adapter_image_rgb888, params->ip_adapter_image_length,
                                params->ip_adapter_image_width, params->ip_adapter_image_height, "IP-Adapter")) return 0;
    if (!std::isfinite(params->control_strength) || params->control_strength < 0.f || params->control_strength > 2.f) {
        setError("ControlNet strength must be finite and within 0..2"); return 0;
    }
    if (!std::isfinite(params->ip_adapter_strength) || params->ip_adapter_strength < 0.f || params->ip_adapter_strength > 2.f) {
        setError("IP-Adapter strength must be finite and within 0..2"); return 0;
    }
    if (params->control_image_rgb888 != nullptr && !handle->has_control_net) {
        setError("ControlNet image selected, but the loaded model set has no control_net weight"); return 0;
    }
    if (params->ip_adapter_image_rgb888 != nullptr && !handle->has_ip_adapter) {
        setError("IP-Adapter reference selected, but the loaded model set has no ip_adapter weight"); return 0;
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
    generation.sample_params.sample_method = static_cast<enum sample_method_t>(params->sample_method);
    generation.sample_params.scheduler = static_cast<enum scheduler_t>(params->scheduler);
    generation.hires.enabled = params->hires_enabled != 0;
    generation.hires.upscaler = static_cast<enum sd_hires_upscaler_t>(params->hires_upscaler);
    generation.hires.model_path = stringIsNonempty(params->hires_model_path) ? params->hires_model_path : nullptr;
    generation.hires.scale = params->hires_scale;
    generation.hires.target_width = params->hires_target_width;
    generation.hires.target_height = params->hires_target_height;
    generation.hires.steps = params->hires_steps;
    generation.hires.denoising_strength = params->hires_denoising_strength;
    generation.hires.upscale_tile_size = params->hires_tile_size;
    generation.control_strength = params->control_strength;
    generation.ip_adapter_strength = params->ip_adapter_strength;
    if (params->control_image_rgb888 != nullptr) {
        generation.control_image.width = static_cast<uint32_t>(params->control_image_width);
        generation.control_image.height = static_cast<uint32_t>(params->control_image_height);
        generation.control_image.channel = 3;
        generation.control_image.data = const_cast<uint8_t*>(params->control_image_rgb888);
    }
    if (params->ip_adapter_image_rgb888 != nullptr) {
        generation.ip_adapter_image.width = static_cast<uint32_t>(params->ip_adapter_image_width);
        generation.ip_adapter_image.height = static_cast<uint32_t>(params->ip_adapter_image_height);
        generation.ip_adapter_image.channel = 3;
        generation.ip_adapter_image.data = const_cast<uint8_t*>(params->ip_adapter_image_rgb888);
    }

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

    sd_image_generation_result_t* image_result = nullptr;
    std::vector<sd_image_t> images_storage;
    sd_image_t* images = nullptr;
    int image_count = 0;
    auto refresh_images = [&]() {
        images_storage.clear();
        images = nullptr;
        image_count = 0;
        if (image_result == nullptr || image_result->items == nullptr || image_result->count <= 0) return;
        images_storage.reserve(static_cast<size_t>(image_result->count));
        for (int i = 0; i < image_result->count; ++i) {
            images_storage.push_back(image_result->items[i].image);
        }
        images = images_storage.data();
        image_count = static_cast<int>(images_storage.size());
    };
    auto release_images = [&]() {
        if (image_result != nullptr) free_sd_image_generation_result(image_result);
        image_result = nullptr;
        images_storage.clear();
        images = nullptr;
        image_count = 0;
    };
    const auto start = std::chrono::steady_clock::now();
    bool ok = generate_image(handle->ctx, &generation, &image_result);
    refresh_images();
    const auto base_generation_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    PONKO_LOGI("BENCHMARK base generation elapsed_ms=%lld", static_cast<long long>(base_generation_ms));
    // Post-generation ADetailer. Cache the detector on this loaded model handle.
    if (image_count > 0 && images != nullptr && images[0].data != nullptr) ok = true;
    if (ok && image_count > 0 && images != nullptr && images[0].data != nullptr &&
        params->adetailer_enabled != 0) {
        const char* detector_path = params->adetailer_detector_path;
        bool detector_ready = false;
        if (!stringIsNonempty(detector_path)) {
            setError("ADetailer is enabled but no YOLOv8 detector file was selected");
        } else {
            detector_ready = handle->adetailer != nullptr &&
                             handle->adetailer_detector_path == detector_path;
            if (!detector_ready) {
                // Use CPU for detection so this does not depend on detector Vulkan kernels.
                const auto detector_load_start = std::chrono::steady_clock::now();
                adetailer_ctx_t* replacement = new_adetailer_ctx(
                    detector_path, handle->n_threads, "cpu", "cpu");
                const auto detector_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - detector_load_start).count();
                PONKO_LOGI("BENCHMARK ADetailer detector load elapsed_ms=%lld success=%d",
                           static_cast<long long>(detector_load_ms), replacement != nullptr ? 1 : 0);
                if (replacement == nullptr) {
                    setError("could not load the ADetailer YOLOv8 detection .safetensors file");
                } else {
                    adetailer_ctx_t* previous = handle->adetailer;
                    handle->adetailer = replacement;
                    handle->adetailer_detector_path = detector_path;
                    detector_ready = true;
                    if (previous != nullptr) free_adetailer_ctx(previous);
                }
            }
        }

        if (!detector_ready) {
            release_images();
            images = nullptr;
            image_count = 0;
            ok = false;
        } else {
            sd_adetailer_params_t detailer_params{};
            detailer_params.prompt = stringIsNonempty(params->adetailer_prompt)
                ? params->adetailer_prompt : nullptr;
            detailer_params.negative_prompt = stringIsNonempty(params->adetailer_negative_prompt)
                ? params->adetailer_negative_prompt : nullptr;
            detailer_params.extra_ad_args = stringIsNonempty(params->adetailer_extra_args)
                ? params->adetailer_extra_args : nullptr;

            sd_img_gen_params_t detail_generation = generation;
            // Match upstream's txt2img post-pass defaults; UI args override these values.
            if (generation.init_image.data == nullptr) {
                detail_generation.width = 512;
                detail_generation.height = 512;
                detail_generation.strength = 0.4f;
            }
            // The full-frame ControlNet map does not align with ADetailer's per-detection crop.
            detail_generation.control_image = {};
            detail_generation.control_strength = 0.0f;
            sd_image_generation_result_t* detailed_result = nullptr;
            std::vector<sd_image_t> detailed_storage;
            const bool detail_ok = adetail_image(
                handle->adetailer, handle->ctx, images[0], &detailer_params,
                &detail_generation, &detailed_result);
            if (detailed_result != nullptr && detailed_result->items != nullptr && detailed_result->count > 0) {
                detailed_storage.reserve(static_cast<size_t>(detailed_result->count));
                for (int i = 0; i < detailed_result->count; ++i) {
                    detailed_storage.push_back(detailed_result->items[i].image);
                }
            }
            sd_image_t* detailed_images = detailed_storage.empty() ? nullptr : detailed_storage.data();
            int detailed_count = static_cast<int>(detailed_storage.size());
            if (!detail_ok || detailed_count <= 0 || detailed_images == nullptr ||
                detailed_images[0].data == nullptr) {
                if (detailed_result != nullptr) free_sd_image_generation_result(detailed_result);
                release_images();
                images = nullptr;
                image_count = 0;
                ok = false;
                setError("ADetailer failed; check detector compatibility and the selected diffusion model");
            } else {
                release_images();
                image_result = detailed_result;
                detailed_result = nullptr;
                images_storage = std::move(detailed_storage);
                images = images_storage.data();
                image_count = static_cast<int>(images_storage.size());
                ok = true;
                PONKO_LOGI("ADetailer completed using detector=%s", detector_path);
            }
        }
    }

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
        if (g_last_error.empty()) setError("stable-diffusion.cpp generation failed");
    }
    release_images();
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
    if (handle->adetailer != nullptr) {
        free_adetailer_ctx(handle->adetailer);
        handle->adetailer = nullptr;
    }
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

extern "C" int32_t ponko_sd_fuse_loras(const char* input_model_path,
                                       const char* const* lora_paths,
                                       const float* lora_scales,
                                       size_t lora_count,
                                       const char* output_path,
                                       const char* quant_type,
                                       int32_t n_threads) {
    g_last_error.clear();
    initializePonkoLogging();
    try {
    PONKO_LOGI("LoRA fusion entered C ABI: adapters=%zu quant=%s threads=%d",
               lora_count, stringIsNonempty(quant_type) ? quant_type : "(null)", n_threads);
    const std::string output_name = stringIsNonempty(output_path) ? output_path : "";
    if (!stringIsNonempty(input_model_path) || output_name.empty() ||
        !stringIsNonempty(quant_type) || lora_paths == nullptr || lora_scales == nullptr ||
        lora_count == 0 || output_name.size() < 5 ||
        output_name.compare(output_name.size() - 5, 5, ".gguf") != 0) {
        setError("LoRA fusion requires an input checkpoint, at least one LoRA, a quantization mode, and a .gguf output path");
        return 0;
    }
    if (std::string(input_model_path) == output_path) {
        setError("LoRA fusion output must be different from the source checkpoint");
        return 0;
    }

    std::ifstream source(input_model_path, std::ios::binary);
    if (!source.good()) {
        setError("LoRA fusion base checkpoint does not exist");
        return 0;
    }
    for (size_t i = 0; i < lora_count; ++i) {
        if (!stringIsNonempty(lora_paths[i]) || !std::isfinite(lora_scales[i])) {
            setError("LoRA fusion contains an empty adapter path or a non-finite scale");
            return 0;
        }
        std::ifstream adapter(lora_paths[i], std::ios::binary);
        if (!adapter.good()) {
            setError(std::string("LoRA fusion adapter file does not exist: ") + lora_paths[i]);
            return 0;
        }
    }
    std::ifstream existing_output(output_path, std::ios::binary);
    if (existing_output.good()) {
        setError("refusing to overwrite an existing LoRA fusion output");
        return 0;
    }

    const bool keep_f16 = std::string(quant_type) == "f16";
    enum sd_type_t quant_type_enum = SD_TYPE_F16;
    std::string rules;
    if (!keep_f16) {
        if (!parseQuantType(quant_type, &quant_type_enum)) return 0;
        rules = std::string("^model\\.diffusion_model\\..*=") + quant_type;
    }

    const std::string temporary = std::string(output_path) + ".partial.gguf";
    std::remove(temporary.c_str());
    const int threads = 1; // Low-RAM device path: serialize fusion tensor workers.
    std::vector<sd_lora_t> loras;
    loras.reserve(lora_count);
    for (size_t i = 0; i < lora_count; ++i) {
        loras.push_back(sd_lora_t{false, lora_scales[i], lora_paths[i]});
    }

    PONKO_LOGI("LoRA fusion entering upstream conversion path: output=%s threads=%d adapters=%zu",
               output_path, threads, loras.size());
    const bool converted = convert_with_components(
        input_model_path, nullptr, nullptr, nullptr, nullptr, nullptr, temporary.c_str(),
        SD_TYPE_F16, keep_f16 ? nullptr : rules.c_str(), true, threads,
        loras.data(), static_cast<int>(loras.size()));
    PONKO_LOGI("LoRA fusion converter returned: success=%d", converted ? 1 : 0);
    if (!converted) {
        std::remove(temporary.c_str());
        setError("native LoRA fusion/export failed; source files were left untouched");
        return 0;
    }

    bool valid = false;
    if (keep_f16) {
        valid = validateReadableGguf(temporary);
    } else {
        int64_t unet_count = 0;
        int64_t quantized_count = 0;
        valid = validateQuantizedUnetFile(temporary, static_cast<enum ggml_type>(quant_type_enum),
                                          &unet_count, &quantized_count);
    }
    if (!valid) {
        std::remove(temporary.c_str());
        return 0;
    }
    if (std::rename(temporary.c_str(), output_path) != 0) {
        std::remove(temporary.c_str());
        setError("failed to atomically publish fused LoRA GGUF output");
        return 0;
    }
    PONKO_LOGI("LoRA fusion complete: adapters=%zu quant=%s output=%s",
               lora_count, quant_type, output_path);
    return 1;
    } catch (const std::exception& e) {
        if (stringIsNonempty(output_path)) {
            const std::string temporary = std::string(output_path) + ".partial.gguf";
            std::remove(temporary.c_str());
        }
        setError(std::string("LoRA fusion caught native exception: ") + e.what());
        PONKO_LOGE("LoRA fusion caught native exception: %s", e.what());
        return 0;
    } catch (...) {
        if (stringIsNonempty(output_path)) {
            const std::string temporary = std::string(output_path) + ".partial.gguf";
            std::remove(temporary.c_str());
        }
        setError("LoRA fusion caught an unknown native exception");
        PONKO_LOGE("LoRA fusion caught an unknown native exception");
        return 0;
    }
}

extern "C" int32_t ponko_sd_quantize_unet(const char* input_path, const char* output_path,
                                           const char* quant_type, int32_t diffusion_slot, int32_t n_threads) {
    g_last_error.clear();
    if (!stringIsNonempty(input_path) || !stringIsNonempty(output_path) || !stringIsNonempty(quant_type)) {
        setError("quantization input, output and type must be provided");
        return 0;
    }
    enum sd_type_t quant_type_enum;
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
    if (!validateQuantizedUnetFile(temporary, static_cast<enum ggml_type>(quant_type_enum),
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
    enum sd_type_t quant_type_enum;
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
    if (!validateQuantizedUnetFile(temporary, static_cast<enum ggml_type>(quant_type_enum),
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

// Convert a PyTorch checkpoint container to safetensors without renaming or quantizing tensors.
extern "C" int32_t ponko_sd_convert_torch_bin_to_safetensors(const char* input_path,
                                                               const char* output_path) {
    g_last_error.clear();
    initializePonkoLogging();
    try {
        if (!stringIsNonempty(input_path) || !stringIsNonempty(output_path)) {
            setError("PyTorch .bin conversion requires input and output paths");
            return 0;
        }
        const std::string input_name(input_path);
        const std::string output_name(output_path);
        constexpr size_t suffix_length = sizeof(".safetensors") - 1;
        if (output_name.size() < suffix_length ||
            output_name.compare(output_name.size() - suffix_length, suffix_length, ".safetensors") != 0) {
            setError("conversion output must use the .safetensors extension");
            return 0;
        }
        if (input_name == output_name) {
            setError("conversion output must be different from the source checkpoint");
            return 0;
        }
        std::ifstream source(input_path, std::ios::binary);
        if (!source.good()) {
            setError("PyTorch .bin source file does not exist or cannot be read");
            return 0;
        }
        std::ifstream existing_output(output_path, std::ios::binary);
        if (existing_output.good()) {
            setError("refusing to overwrite an existing safetensors output");
            return 0;
        }
        // Keep the staging suffix recognizable to the converter's format-selecting writer.
        const std::string temporary = output_name + ".partial.safetensors";
        std::remove(temporary.c_str());
        // On Android, avoid parallel tensor export workers for this conversion. Each worker
        // can hold a source ZIP storage buffer, the destination tensor, and a serialized copy
        // at the same time. The generic convert() wrapper selects all physical cores.
        PONKO_LOGI("PyTorch .bin conversion start: input=%s output=%s writer_threads=1",
                   input_path, output_path);
        const bool converted = convert_with_components(
            input_path, nullptr, nullptr, nullptr, nullptr, nullptr,
            temporary.c_str(), SD_TYPE_COUNT, nullptr, false, 1);
        PONKO_LOGI("PyTorch .bin streaming conversion returned: success=%d", converted ? 1 : 0);
        if (!converted) {
            std::remove(temporary.c_str());
            setError("stable-diffusion.cpp could not convert this PyTorch checkpoint; inspect native logs for the unsupported format or tensor");
            return 0;
        }
        PONKO_LOGI("PyTorch .bin conversion validating safetensors output");
        std::vector<TensorStorage> tensors;
        std::string validation_error;
        const bool valid = is_safetensors_file(temporary) &&
                           read_safetensors_file(temporary, tensors, &validation_error) &&
                           !tensors.empty();
        PONKO_LOGI("PyTorch .bin safetensors validation returned: valid=%d tensors=%zu",
                   valid ? 1 : 0, tensors.size());
        if (!valid) {
            std::remove(temporary.c_str());
            setError(validation_error.empty()
                         ? "converted safetensors output failed validation or contains no tensors"
                         : "converted safetensors output failed validation: " + validation_error);
            return 0;
        }
        if (std::rename(temporary.c_str(), output_path) != 0) {
            std::remove(temporary.c_str());
            setError("failed to publish the validated safetensors output");
            return 0;
        }
        PONKO_LOGI("PyTorch checkpoint conversion complete: tensors=%zu output=%s", tensors.size(), output_path);
        return 1;
    } catch (const std::exception& e) {
        setError(std::string("PyTorch .bin conversion caught native exception: ") + e.what());
        PONKO_LOGE("PyTorch .bin conversion exception: %s", e.what());
        return 0;
    } catch (...) {
        setError("PyTorch .bin conversion caught an unknown native exception");
        PONKO_LOGE("PyTorch .bin conversion caught an unknown native exception");
        return 0;
    }
}
