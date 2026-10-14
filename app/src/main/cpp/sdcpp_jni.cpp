// Ponko · (c) 2026 visitor257 · MIT License · github.com/visitor257/Ponko
// sdcpp_jni.cpp — Ponko 自编 stable-diffusion.cpp 的 JNI 桥
// 目标：把 sd.cpp 的 C API 暴露给 Kotlin（可控线程数 / 采样器 / 调度器 / LoRA / 量化类型 / 取消）
#include <jni.h>
#include <android/log.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>
#include <cstdio>

#include "ggml.h"
#include "ggml-vulkan.h"
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
    const std::string logged = copyVulkanDiagnostics();
    if (!logged.empty()) {
        if (!diagnostics.empty()) diagnostics.push_back('\n');
        diagnostics += logged;
    }
    return env->NewStringUTF(diagnostics.c_str());
}

// Recent native configuration is stored for writing into Kotlin's diagnostic log.
extern "C" JNIEXPORT jstring JNICALL
Java_com_litertchat_app_draw_SdCppEngine_nativeLastConfig(JNIEnv* env, jobject /*thiz*/) {
    return env->NewStringUTF(g_last_dump.c_str());
}
