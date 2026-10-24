// Ponko · (c) 2026 visitor257 · MIT License · github.com/visitor257/Ponko
// sdcpp_jni.cpp — Ponko 自编 stable-diffusion.cpp 的 JNI 桥
// 目标：把 sd.cpp 的 C API 暴露给 Kotlin（可控线程数 / 采样器 / 调度器 / LoRA / 量化类型 / 取消）
#include <jni.h>
#include <android/log.h>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>
#include <cstdio>
#include <chrono>

#include "ggml.h"
#include "ggml-vulkan.h"
#include "gguf.h"
#include "stable-diffusion.h"

#define TAG "PonkoSd"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

JavaVM* g_vm = nullptr;

struct SdHandle {
    sd_ctx_t* ctx = nullptr;
    jobject   cb  = nullptr;    // StepCallback（GlobalRef）
    jmethodID cb_mid = nullptr; // onStep(II)V
    jobject   pcb = nullptr;    // PreviewCallback（GlobalRef）
    jmethodID pcb_mid = nullptr; // onPreview(III[B)V
    std::atomic<bool> cancelled{false};
    bool vae_conv_direct = true;
    std::string backend_name = "auto";
};

SdHandle* g_active = nullptr;  // 同时只允许一个生成（进度回调是全局的）

// 最近一次 new_sd_ctx 的参数字符串（供 Kotlin 写进日志文件，方便定位加载失败）
std::string g_last_dump;

std::mutex g_vulkan_diagnostics_mutex;
std::string g_vulkan_diagnostics;

void ponkoGgmlLogCallback(enum ggml_log_level level, const char* text, void* /*user_data*/) {
    if (text == nullptr || text[0] == '\0') return;

    const bool is_vulkan = std::strstr(text, "ggml_vulkan:") != nullptr ||
                           std::strstr(text, "Xclipse 940") != nullptr;
    const bool is_warning = level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR;

    // Keep the original stderr behavior, and additionally surface relevant lines in Logcat.
    std::fputs(text, stderr);
    std::fflush(stderr);
    if (is_vulkan || is_warning) {
        const int priority = level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR :
                             level == GGML_LOG_LEVEL_WARN  ? ANDROID_LOG_WARN  :
                             level == GGML_LOG_LEVEL_DEBUG ? ANDROID_LOG_DEBUG : ANDROID_LOG_INFO;
        __android_log_print(priority, is_vulkan ? "PonkoVulkan" : "PonkoSd", "%s", text);
    }

    const bool is_capability_line =
        std::strstr(text, "Xclipse 940 profile detected") != nullptr ||
        std::strstr(text, "matrix cores:") != nullptr;
    if (is_capability_line) {
        std::lock_guard<std::mutex> lock(g_vulkan_diagnostics_mutex);
        if (g_vulkan_diagnostics.find(text) == std::string::npos) {
            g_vulkan_diagnostics.append(text);
            if (g_vulkan_diagnostics.size() > 8192) {
                g_vulkan_diagnostics.erase(0, g_vulkan_diagnostics.size() - 8192);
            }
        }
    }
}

std::string copyVulkanDiagnostics() {
    std::lock_guard<std::mutex> lock(g_vulkan_diagnostics_mutex);
    return g_vulkan_diagnostics;
}

const char* jstr(JNIEnv* env, jstring s, std::string& holder) {
    if (s == nullptr) return nullptr;
    const char* c = env->GetStringUTFChars(s, nullptr);
    holder.assign(c ? c : "");
    env->ReleaseStringUTFChars(s, c);
    return holder.c_str();
}

// 进度回调：由 native 推理线程调用 → 需要 attach 到 JVM
void progressTrampoline(int step, int steps, float /*time*/, void* data) {
    SdHandle* h = static_cast<SdHandle*>(data);
    if (!h || !h->cb || !h->cb_mid) return;
    if (h->cancelled.load()) return;

    JNIEnv* env = nullptr;
    bool attached = false;
    if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK || env == nullptr) return;
        attached = true;
    }
    env->CallVoidMethod(h->cb, h->cb_mid, static_cast<jint>(step), static_cast<jint>(steps));
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (attached) g_vm->DetachCurrentThread();
}

// 实时预览回调：native 推理线程每步调用一次。frames 指向的内存回调返回后立即释放，
// 必须当场拷成 jbyteArray 抛给 Kotlin。
void previewTrampoline(int step, int frame_count, sd_image_t* frames, bool /*is_noisy*/, void* data) {
    SdHandle* h = static_cast<SdHandle*>(data);
    if (!h || !h->pcb || !h->pcb_mid) return;
    if (h->cancelled.load()) return;
    if (frames == nullptr || frame_count <= 0) return;
    const sd_image_t& im = frames[0];
    if (im.data == nullptr || im.width == 0 || im.height == 0 || im.channel != 3) return;

    JNIEnv* env = nullptr;
    bool attached = false;
    if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK || env == nullptr) return;
        attached = true;
    }
    const jsize n = static_cast<jsize>(im.width * im.height * 3u);
    jbyteArray arr = env->NewByteArray(n);
    if (arr != nullptr) {
        env->SetByteArrayRegion(arr, 0, n, reinterpret_cast<const jbyte*>(im.data));
        env->CallVoidMethod(h->pcb, h->pcb_mid, static_cast<jint>(step),
                            static_cast<jint>(im.width), static_cast<jint>(im.height), arr);
        env->DeleteLocalRef(arr);
    }
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (attached) g_vm->DetachCurrentThread();
}


// Validate the temporary GGUF before publishing it. This catches truncated/corrupt output,
// proves that the diffusion tensors really received the requested quant type, and rejects
// accidental low-bit quantization outside the model.diffusion_model.* namespace.
bool validateQuantizedUnetFile(const std::string& path,
                               ggml_type expectedType,
                               long long* unetTensorCount,
                               long long* quantizedTensorCount) {
    if (unetTensorCount) *unetTensorCount = 0;
    if (quantizedTensorCount) *quantizedTensorCount = 0;

    gguf_init_params params = { true, nullptr };
    gguf_context* gguf = gguf_init_from_file(path.c_str(), params);
    if (gguf == nullptr) {
        LOGE("UNet quantizer: output is not a readable GGUF file");
        return false;
    }

    const char* prefix = "model.diffusion_model.";
    const size_t prefixLength = std::strlen(prefix);
    long long unetCount = 0;
    long long quantCount = 0;
    long long outsideQuantCount = 0;
    const int64_t tensorCount = gguf_get_n_tensors(gguf);
    for (int64_t i = 0; i < tensorCount; ++i) {
        const char* name = gguf_get_tensor_name(gguf, i);
        if (name == nullptr) continue;
        const bool isUnet = std::strncmp(name, prefix, prefixLength) == 0;
        const ggml_type type = gguf_get_tensor_type(gguf, i);
        if (isUnet) {
            ++unetCount;
            if (type == expectedType) ++quantCount;
        } else if (type == expectedType) {
            ++outsideQuantCount;
        }
    }
    gguf_free(gguf);

    if (unetTensorCount) *unetTensorCount = unetCount;
    if (quantizedTensorCount) *quantizedTensorCount = quantCount;
    if (unetCount == 0 || quantCount == 0 || outsideQuantCount != 0) {
        LOGE("UNet quantizer: validation failed (unet=%lld, requested_type=%lld, outside_unet=%lld)",
             unetCount, quantCount, outsideQuantCount);
        return false;
    }
    return true;
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* /*reserved*/) {
    g_vm = vm;
    ggml_log_set(ponkoGgmlLogCallback, nullptr);
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_litertchat_app_draw_SdCppEngine_nativeVulkanDiagnostics(JNIEnv* env, jobject /*thiz*/) {
    // Query ggml-vulkan's live device state directly. Log callbacks can be routed through a
    // different ggml logger instance in shared-library builds, leaving the old report empty.
    char deviceDiagnostics[2048] = {};
    const int deviceCount = ggml_backend_vk_get_device_count();
    if (deviceCount > 0) {
        ggml_backend_vk_get_device_diagnostics(0, deviceDiagnostics, sizeof(deviceDiagnostics));
    } else {
        std::snprintf(deviceDiagnostics, sizeof(deviceDiagnostics), "No Vulkan devices enumerated by ggml-vulkan");
    }

    std::string diagnostics(deviceDiagnostics);
    const std::string loggedDiagnostics = copyVulkanDiagnostics();
    if (!loggedDiagnostics.empty() && diagnostics.find(loggedDiagnostics) == std::string::npos) {
        if (!diagnostics.empty()) diagnostics += "\n";
        diagnostics += loggedDiagnostics;
    }
    return env->NewStringUTF(diagnostics.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_litertchat_app_draw_SdCppEngine_nativeInfo(JNIEnv* env, jobject /*thiz*/) {
    std::string info = std::string("sd.cpp ") + sd_version() + " | " + sd_get_system_info();
    return env->NewStringUTF(info.c_str());
}

// 最近一次 new_sd_ctx 用的完整参数（失败时写进日志文件用）
extern "C" JNIEXPORT jstring JNICALL
Java_com_litertchat_app_draw_SdCppEngine_nativeLastParams(JNIEnv* env, jobject /*thiz*/) {
    return env->NewStringUTF(g_last_dump.c_str());
}

// nativeListDevices() -> "name\tdescription\n..."（ggml 后端设备清单；没有设备时返回空串）
extern "C" JNIEXPORT jstring JNICALL
Java_com_litertchat_app_draw_SdCppEngine_nativeListDevices(JNIEnv* env, jobject /*thiz*/) {
    const size_t need = sd_list_devices(nullptr, 0);
    if (need == 0) return env->NewStringUTF("");
    std::vector<char> buf(need + 1, '\0');
    sd_list_devices(buf.data(), buf.size());
    return env->NewStringUTF(buf.data());
}

// 槽位顺序（与 Kotlin 侧 SdCppEngine.SLOT_KEYS 一一对应）
enum SlotIndex {
    SLOT_MODEL = 0,       // 单文件 all-in-one -> model_path
    SLOT_DIFFUSION,       // DiT/UNet 单独文件 -> diffusion_model_path
    SLOT_HIGH_NOISE,      // 高噪扩散模型
    SLOT_UNCOND,          // 无条件扩散模型
    SLOT_CLIP_L,
    SLOT_CLIP_G,
    SLOT_CLIP_VISION,
    SLOT_T5XXL,
    SLOT_LLM,
    SLOT_LLM_VISION,
    SLOT_VAE,
    SLOT_AUDIO_VAE,
    SLOT_TAESD,
    SLOT_COUNT
};

// nativeCreate(modelPath, vaePath, nThreads, wtype, flashAttn, backend, slots) -> handle
//
// backend：ggml 后端/设备名（如 "CPU" / "Vulkan0" / "vulkan"），空串或 null = 交给 sd.cpp 自动挑。
// 显式传值会关掉 sd.cpp 的 auto_fit，行为更可控（选 CPU 就真是 CPU）。
//
// slots：模型槽位路径数组，按 SlotIndex 顺序给槽位文件路径（当前 13 项）；
// 元素可以为 null / 空串（表示该槽位不填）。传 null = 走旧的两参数路径（单文件 + 可选 VAE）。
extern "C" JNIEXPORT jlong JNICALL
Java_com_litertchat_app_draw_SdCppEngine_nativeCreate(
        JNIEnv* env, jobject /*thiz*/,
        jstring modelPath, jstring vaePath, jint nThreads, jint wtype, jboolean flashAttn,
        jboolean vaeConvDirect, jstring backend, jobjectArray slots) {
    std::string model, vae, backendStr;
    jstr(env, modelPath, model);
    jstr(env, vaePath, vae);
    jstr(env, backend, backendStr);
    if (modelPath == nullptr && slots == nullptr) return 0;

    // 槽位路径（std::string 持有内容，new_sd_ctx 期间必须有效）
    std::vector<std::string> slotStr(SLOT_COUNT);
    std::vector<const char*> slotPtr(SLOT_COUNT, nullptr);
    if (slots != nullptr) {
        const jsize n = env->GetArrayLength(slots);
        for (jsize i = 0; i < n && i < SLOT_COUNT; ++i) {
            jstring js = static_cast<jstring>(env->GetObjectArrayElement(slots, i));
            if (js == nullptr) continue;
            const char* cs = env->GetStringUTFChars(js, nullptr);
            if (cs != nullptr && cs[0] != '\0') {
                slotStr[i] = cs;
                slotPtr[i] = slotStr[i].c_str();
            }
            if (cs != nullptr) env->ReleaseStringUTFChars(js, cs);
            env->DeleteLocalRef(js);
        }
        bool any = false;
        for (int i = 0; i < SLOT_COUNT; ++i) if (slotPtr[i] != nullptr) any = true;
        if (!any) return 0;
    }

    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    const char* modelPathC = slotPtr[SLOT_MODEL] != nullptr ? slotPtr[SLOT_MODEL]
                           : (model.empty() ? nullptr : model.c_str());
    const char* vaePathC   = slotPtr[SLOT_VAE] != nullptr ? slotPtr[SLOT_VAE]
                           : (vae.empty() ? nullptr : vae.c_str());
    p.model_path            = modelPathC;
    p.diffusion_model_path  = slotPtr[SLOT_DIFFUSION];
    p.high_noise_diffusion_model_path = slotPtr[SLOT_HIGH_NOISE];
    p.uncond_diffusion_model_path     = slotPtr[SLOT_UNCOND];
    p.clip_l_path           = slotPtr[SLOT_CLIP_L];
    p.clip_g_path           = slotPtr[SLOT_CLIP_G];
    p.clip_vision_path      = slotPtr[SLOT_CLIP_VISION];
    p.t5xxl_path            = slotPtr[SLOT_T5XXL];
    p.llm_path              = slotPtr[SLOT_LLM];
    p.llm_vision_path       = slotPtr[SLOT_LLM_VISION];
    p.vae_path              = vaePathC;
    p.audio_vae_path        = slotPtr[SLOT_AUDIO_VAE];
    p.taesd_path             = slotPtr[SLOT_TAESD];
    p.n_threads    = (nThreads > 0) ? nThreads : sd_get_num_physical_cores();
    // wtype < 0 表示「保持模型原样」——此时【不能】覆盖，保留 sd_ctx_params_init 给的
    // SD_TYPE_COUNT。实测把 -1 强转成 sd_type_t 会让 sd.cpp 内部查表越界，native 直接崩溃。
    if (wtype >= 0) p.wtype = static_cast<enum sd_type_t>(wtype);
    p.enable_mmap  = true;
    // 后端：非空则显式指定（同时关掉 auto_fit），空则保持 sd.cpp 默认（自动挑设备）
    if (!backendStr.empty()) p.backend = backendStr.c_str();
    // 两个 FlashAttention 都要开：flash_attn 影响 CLIP，diffusion_flash_attn 影响 UNet（算力主体）。
    // 实测不开比开慢约 28%（6 步 125s vs 98s）。注意 sd_ctx_params_init 默认都是 false。
    const bool fa = (flashAttn == JNI_TRUE);
    p.flash_attn           = fa;
    p.diffusion_flash_attn = fa;
    // Expose the direct-convolution experiment as an explicit runtime option.
    p.vae_conv_direct = (vaeConvDirect == JNI_TRUE);
    p.lora_apply_mode = LORA_APPLY_AUTO;

    char* dump = sd_ctx_params_to_str(&p);
    g_last_dump = dump ? dump : "";
    LOGI("new_sd_ctx: %s", dump ? dump : "(null)");
    if (dump) free(dump);

    sd_ctx_t* ctx = new_sd_ctx(&p);
    if (ctx == nullptr) {
        LOGE("new_sd_ctx returned null");
        return 0;
    }
    SdHandle* h = new SdHandle();
    h->ctx = ctx;
    h->vae_conv_direct = p.vae_conv_direct;
    h->backend_name = backendStr.empty() ? "auto" : backendStr;
    LOGI("ctx created: %p | VAE direct-conv=%d | backend=%s",
         static_cast<void*>(ctx), h->vae_conv_direct ? 1 : 0, h->backend_name.c_str());
    return reinterpret_cast<jlong>(h);
}

// nativeGenerate(handle, prompt, negative, loraPath, loraScale,
//                width, height, steps, cfg, seed, sampleMethod, scheduler, clipSkip,
//                cb, initData, initWidth, initHeight, strength) -> IntArray(ARGB)
//
// initData 非空 → 图生图（img2img）：RGB888 交错字节（每像素 3 字节，行优先，左上原点），
// sd.cpp 会把它按 request 尺寸缩放后编码进 latent，并按 strength(<1) 决定起始噪声步数。
// initData 为空 → 文生图，strength 忽略。
extern "C" JNIEXPORT jintArray JNICALL
Java_com_litertchat_app_draw_SdCppEngine_nativeGenerate(
        JNIEnv* env, jobject /*thiz*/,
        jlong handle,
        jstring prompt, jstring negative,
        jobjectArray loraPaths, jfloatArray loraScales,
        jint width, jint height, jint steps, jfloat cfg, jlong seed,
        jint sampleMethod, jint scheduler, jint clipSkip,
        jobject cb, jobject pcb,
        jbyteArray initData, jint initWidth, jint initHeight, jfloat strength) {
    SdHandle* h = reinterpret_cast<SdHandle*>(handle);
    if (h == nullptr || h->ctx == nullptr) return nullptr;

    std::string sPrompt, sNeg;
    jstr(env, prompt, sPrompt);
    jstr(env, negative, sNeg);

    const jsize pathCount = loraPaths == nullptr ? 0 : env->GetArrayLength(loraPaths);
    const jsize scaleCount = loraScales == nullptr ? 0 : env->GetArrayLength(loraScales);
    const jsize loraInputCount = pathCount < scaleCount ? pathCount : scaleCount;
    std::vector<std::string> sLoras(static_cast<size_t>(loraInputCount));
    std::vector<float> loraMultipliers(static_cast<size_t>(loraInputCount), 1.0f);
    if (loraInputCount > 0) {
        env->GetFloatArrayRegion(loraScales, 0, loraInputCount, loraMultipliers.data());
        for (jsize i = 0; i < loraInputCount; ++i) {
            auto js = static_cast<jstring>(env->GetObjectArrayElement(loraPaths, i));
            jstr(env, js, sLoras[static_cast<size_t>(i)]);
            if (js != nullptr) env->DeleteLocalRef(js);
        }
    }
    std::vector<sd_lora_t> nativeLoras;
    nativeLoras.reserve(sLoras.size());
    for (size_t i = 0; i < sLoras.size(); ++i) {
        if (sLoras[i].empty()) continue;
        sd_lora_t entry{};
        entry.is_high_noise = false;
        entry.multiplier = loraMultipliers[i];
        entry.path = sLoras[i].c_str();
        nativeLoras.push_back(entry);
    }

    // 进度回调（全局注册，data 指向本 handle）
    if (cb != nullptr) {
        h->cb = env->NewGlobalRef(cb);
        jclass cls = env->GetObjectClass(cb);
        h->cb_mid = env->GetMethodID(cls, "onStep", "(II)V");
        if (h->cb_mid == nullptr) {
            LOGE("StepCallback.onStep(II)V not found");
            env->DeleteGlobalRef(h->cb);
            h->cb = nullptr;
        }
    }
    // 预览回调：每步把 latent 投影成小图（PREVIEW_PROJ 不需要额外模型）
    if (pcb != nullptr) {
        h->pcb = env->NewGlobalRef(pcb);
        jclass pcls = env->GetObjectClass(pcb);
        h->pcb_mid = env->GetMethodID(pcls, "onPreview", "(III[B)V");
        if (h->pcb_mid == nullptr) {
            LOGE("PreviewCallback.onPreview(III[B)V not found");
            env->DeleteGlobalRef(h->pcb);
            h->pcb = nullptr;
        } else {
            sd_set_preview_callback(previewTrampoline, PREVIEW_PROJ, 1, true, false, h);
        }
    } else {
        sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, true, false, nullptr);
    }
    h->cancelled.store(false);
    g_active = h;
    sd_set_progress_callback(progressTrampoline, h);

    sd_img_gen_params_t g;
    sd_img_gen_params_init(&g);
    g.prompt          = sPrompt.c_str();
    g.negative_prompt = sNeg.empty() ? nullptr : sNeg.c_str();
    g.width           = width;
    g.height          = height;
    g.seed            = seed;
    g.batch_count     = 1;
    g.strength        = 1.0f;
    g.clip_skip       = (clipSkip < 1) ? 1 : ((clipSkip > 12) ? 12 : clipSkip);

    // 图生图：拿到 RGB888 原始字节，指针在 generate_image 期间必须保持有效。
    // 生成结束（含失败）后统一在下方 ReleaseByteArrayElements 释放。
    jbyte* initBytes = nullptr;
    if (initData != nullptr && initWidth > 0 && initHeight > 0) {
        initBytes = env->GetByteArrayElements(initData, nullptr);
        if (initBytes != nullptr) {
            g.init_image.width   = static_cast<uint32_t>(initWidth);
            g.init_image.height  = static_cast<uint32_t>(initHeight);
            g.init_image.channel = 3;
            g.init_image.data    = reinterpret_cast<uint8_t*>(initBytes);
            // strength 夹到 (0,1)：=1 会退化成「全部重绘」（等价文生图，但白跑一遍 VAE 编码）
            float s = strength;
            if (!(s > 0.f)) s = 0.01f;
            if (s >= 1.f)   s = 0.99f;
            g.strength = s;
            LOGI("img2img: init %dx%d, strength=%.2f", initWidth, initHeight, s);
        } else {
            LOGE("img2img: GetByteArrayElements returned null");
        }
    }
    g.loras = nativeLoras.empty() ? nullptr : nativeLoras.data();
    g.lora_count = static_cast<uint32_t>(nativeLoras.size());
    g.sample_params.sample_steps  = steps;
    g.sample_params.guidance.txt_cfg = cfg;
    g.sample_params.sample_method = static_cast<enum sample_method_t>(sampleMethod);
    g.sample_params.scheduler     = static_cast<enum scheduler_t>(scheduler);

    char* gdump = sd_img_gen_params_to_str(&g);
    LOGI("generate_image: %s", gdump ? gdump : "(null)");
    if (gdump) free(gdump);

    sd_image_t* out = nullptr;
    int n = 0;
    const auto generationStart = std::chrono::steady_clock::now();
    bool ok = generate_image(h->ctx, &g, &out, &n);
    const auto generationElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - generationStart).count();
    if (n > 0 && out != nullptr && out->data != nullptr) ok = true;
    LOGI("BENCHMARK generate_image elapsed_ms=%lld mode=%s size=%dx%d steps=%d backend=%s vae_conv_direct=%d ok=%d",
         static_cast<long long>(generationElapsedMs),
         initBytes != nullptr ? "img2img" : "txt2img",
         width, height, steps, h->backend_name.c_str(),
         h->vae_conv_direct ? 1 : 0, ok ? 1 : 0);

    jintArray result = nullptr;
    if (ok && n > 0 && out != nullptr && out[0].data != nullptr) {
        const sd_image_t& img = out[0];
        const int w = static_cast<int>(img.width);
        const int hh = static_cast<int>(img.height);
        const int ch = static_cast<int>(img.channel);
        const int total = w * hh;
        std::vector<jint> px(static_cast<size_t>(total));
        for (int i = 0; i < total; ++i) {
            const uint8_t* s = img.data + static_cast<size_t>(i) * ch;
            jint r, gg, b;
            if (ch >= 3) { r = s[0]; gg = s[1]; b = s[2]; }
            else         { r = gg = b = s[0]; }
            px[static_cast<size_t>(i)] =
                static_cast<jint>(0xFF000000u | (static_cast<uint32_t>(r) << 16) |
                                  (static_cast<uint32_t>(gg) << 8) | static_cast<uint32_t>(b));
        }
        result = env->NewIntArray(total);
        env->SetIntArrayRegion(result, 0, total, px.data());
        free_sd_images(out, n);
    } else {
        if (out) free_sd_images(out, n);
        LOGE("generate_image failed (ok=%d, n=%d)", ok ? 1 : 0, n);
    }

    if (initBytes != nullptr && initData != nullptr) {
        env->ReleaseByteArrayElements(initData, initBytes, JNI_ABORT);
    }
    if (h->cb) { env->DeleteGlobalRef(h->cb); h->cb = nullptr; h->cb_mid = nullptr; }
    if (h->pcb) { env->DeleteGlobalRef(h->pcb); h->pcb = nullptr; h->pcb_mid = nullptr; }
    sd_set_progress_callback(nullptr, nullptr);
    sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, true, false, nullptr);
    g_active = nullptr;
    return result;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_litertchat_app_draw_SdCppEngine_nativeQuantizeUnet(
        JNIEnv* env, jobject /*thiz*/,
        jstring inputPath, jstring outputPath, jstring quantType,
        jboolean diffusionSlot, jint nThreads) {
    std::string input, output, typeName;
    jstr(env, inputPath, input);
    jstr(env, outputPath, output);
    jstr(env, quantType, typeName);
    if (input.empty() || output.empty()) {
        LOGE("UNet quantizer: input/output path is empty");
        return JNI_FALSE;
    }

    enum sd_type_t quantTypeEnum;
    if (typeName == "q4_0") {
        quantTypeEnum = SD_TYPE_Q4_0;
    } else if (typeName == "q5_0") {
        quantTypeEnum = SD_TYPE_Q5_0;
    } else if (typeName == "q8_0") {
        quantTypeEnum = SD_TYPE_Q8_0;
    } else {
        LOGE("UNet quantizer: unsupported quant type '%s'", typeName.c_str());
        return JNI_FALSE;
    }

    std::ifstream sourceCheck(input, std::ios::binary);
    if (!sourceCheck.good()) {
        LOGE("UNet quantizer: source file does not exist");
        return JNI_FALSE;
    }
    sourceCheck.close();

    std::ifstream outputCheck(output, std::ios::binary);
    if (outputCheck.good()) {
        LOGE("UNet quantizer: refusing to overwrite an existing output file");
        return JNI_FALSE;
    }
    outputCheck.close();

    // Keep the temporary extension as .gguf so the converter selects GGUF output.
    const std::string temporary = output + ".partial.gguf";
    std::remove(temporary.c_str());

    // The default type is F16. Only model.diffusion_model.* tensors are overridden to
    // the selected low-bit type. For a split model, the converter adds that prefix to the
    // standalone diffusion/UNet file; for a single-file checkpoint it already exists.
    const std::string typeRules = std::string("^model\\.diffusion_model\\.=") + typeName;
    const char* modelPath = diffusionSlot == JNI_TRUE ? nullptr : input.c_str();
    const char* diffusionPath = diffusionSlot == JNI_TRUE ? input.c_str() : nullptr;
    const int threads = nThreads < 1 ? 1 : (nThreads > 6 ? 6 : nThreads);

    LOGI("UNet quantizer: source=%s, slot=%s, type=%s, threads=%d",
         input.c_str(), diffusionSlot == JNI_TRUE ? "diffusion" : "model",
         typeName.c_str(), threads);

    const bool converted = convert_with_components(
        modelPath, nullptr, nullptr, nullptr, diffusionPath, nullptr,
        temporary.c_str(), SD_TYPE_F16, typeRules.c_str(), false, threads);
    if (!converted) {
        std::remove(temporary.c_str());
        LOGE("UNet quantizer: stable-diffusion.cpp conversion failed; source left untouched");
        return JNI_FALSE;
    }

    long long unetTensors = 0;
    long long quantizedTensors = 0;
    const bool valid = validateQuantizedUnetFile(
        temporary, static_cast<ggml_type>(quantTypeEnum), &unetTensors, &quantizedTensors);
    if (!valid) {
        std::remove(temporary.c_str());
        LOGE("UNet quantizer: temporary GGUF rejected; source left untouched");
        return JNI_FALSE;
    }

    // Publish only after the converter and the GGUF metadata/type checks have succeeded.
    // The temporary and final files live in the same directory, so rename is atomic on Android.
    if (std::rename(temporary.c_str(), output.c_str()) != 0) {
        std::remove(temporary.c_str());
        LOGE("UNet quantizer: failed to publish output file");
        return JNI_FALSE;
    }

    LOGI("UNet quantizer: success, UNet tensors=%lld, requested-type tensors=%lld, output=%s",
         unetTensors, quantizedTensors, output.c_str());
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_litertchat_app_draw_SdCppEngine_nativeCancel(JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    SdHandle* h = reinterpret_cast<SdHandle*>(handle);
    if (h == nullptr) return;
    h->cancelled.store(true);
    if (h->ctx != nullptr) sd_cancel_generation(h->ctx, SD_CANCEL_ALL);
    LOGI("cancel requested");
}

extern "C" JNIEXPORT void JNICALL
Java_com_litertchat_app_draw_SdCppEngine_nativeFree(JNIEnv* env, jobject /*thiz*/, jlong handle) {
    SdHandle* h = reinterpret_cast<SdHandle*>(handle);
    if (h == nullptr) return;
    if (h->ctx != nullptr) free_sd_ctx(h->ctx);
    if (h->cb != nullptr) env->DeleteGlobalRef(h->cb);
    if (h->pcb != nullptr) env->DeleteGlobalRef(h->pcb);
    delete h;
    LOGI("ctx freed");
}
