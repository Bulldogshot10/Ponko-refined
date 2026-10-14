#!/usr/bin/env python3
"""Source-level contract checks for the TAESD and CPU-thread controls."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}", file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS: {message}")


draw = read("app/src/main/java/com/litertchat/app/draw/DrawPage.kt")
engine = read("app/src/main/java/com/litertchat/app/draw/SdCppEngine.kt")
activity = read("app/src/main/java/com/litertchat/app/MainActivity.kt")
jni = read("app/src/main/cpp/sdcpp_jni.cpp")
header = read("app/src/main/cpp/sd/include/stable-diffusion.h")
diffusion = read("app/src/main/cpp/sd/src/pipeline/diffusion_engine.cpp")
builders = read("app/src/main/cpp/sd/src/pipeline/model_builders.cpp")
vulkan = read("app/src/main/cpp/sd/ggml/src/ggml-vulkan/ggml-vulkan.cpp")
cmake = read("app/src/main/cpp/CMakeLists.txt")
workflow = read(".github/workflows/ponko-controls-validation.yml")

# The slot order is an ABI contract between Kotlin and C++.
keys_match = re.search(r"val SLOT_KEYS = arrayOf\((.*?)\)", engine, re.S)
require(keys_match is not None, "Kotlin SLOT_KEYS declaration exists")
kotlin_keys = re.findall(r'"([a-z0-9_]+)"', keys_match.group(1))

enum_match = re.search(r"enum SlotIndex\s*\{(.*?)\};", jni, re.S)
require(enum_match is not None, "JNI SlotIndex declaration exists")
native_slots = []
for name in re.findall(r"\b(SLOT_[A-Z0-9_]+)\b", enum_match.group(1)):
    if name == "SLOT_COUNT":
        continue
    native_slots.append(name.removeprefix("SLOT_").lower())

require(kotlin_keys == native_slots and len(kotlin_keys) == 13,
        f"Kotlin/native slot order matches all {len(kotlin_keys)} entries")

require('"taesd" -> R.string.draw_taesd_label' in draw,
        "TAESD slot has a UI label")
require('kind == KIND_SINGLE && family == FAM_SD15 -> listOf("model" to true, "vae" to false, "taesd" to false)' in draw,
        "TAESD is optional for single-file SD1.5 models")
require('fun taesdEnabled(): Boolean' in draw and 'fun setTaesdEnabled(enabled: Boolean)' in draw,
        "TAESD has an explicit persisted opt-in setting")
require('if (!taesdEnabled()) slots.remove("taesd")' in draw,
        "unchecked TAESD setting removes the native TAESD path")
require('cur.dir == null -> {' in draw and 'LinkedHashMap(cur.files)' in draw,
        "adding TAESD to a legacy model preserves its main model")
require('p.taesd_path             = slotPtr[SLOT_TAESD];' in jni,
        "JNI forwards the selected TAESD file to sd_ctx_params_t")
require("const char* taesd_path;" in header,
        "bundled native API exposes taesd_path")
require('if (strlen(SAFE_STR(sd_ctx_params->taesd_path)) > 0)' in diffusion,
        "native model loader checks the TAESD path")
require('use_tae = true;' in diffusion,
        "successful TAESD load activates the TAE pipeline")
require('entry.first.rfind("tae.decoder.layers.", 0) == 0' in diffusion,
        "native loader rejects TAESD files without compatible decoder tensors")
require('if (options.use_tae && !options.tae_preview_only)' in builders,
        "TAE can replace the final VAE decoder, not only preview")
require('entry.first.rfind("tae.encoder.layers.", 0) == 0' in builders,
        "native builder detects whether encoder weights are present")
require('TAESD encoder tensors not present; using decoder-only TAE' in builders,
        "decoder-only TAESD skips the unavailable encoder")
tae_branch = builders.index('} else if (options.use_tae && !options.tae_preview_only) {')
standard_vae_branch = builders.index('LOG_INFO("using VAE for encoding / decoding")', tae_branch)
require(tae_branch < standard_vae_branch,
        "TAE opt-in branch is exclusive and selected before the standard VAE branch")

# CPU selection is persisted, bounded, and carried all the way to sd.cpp.
require('private const val KEY_CPU_THREADS = "drawCpuThreads"' in draw,
        "CPU-thread preference has a stable key")
require('prefs().getInt(KEY_CPU_THREADS, 4)' in draw,
        "CPU-thread selection is restored from preferences")
require('nThreads = value.coerceIn(1, maxThreads)' in draw,
        "CPU-thread selection is bounded")
require('putInt(KEY_CPU_THREADS, nThreads)' in draw,
        "CPU-thread selection is persisted")
require('drawPage?.setCpuThreads(threadOptions[position].toInt())' in activity,
        "thread selector writes the chosen value")
require('nThreads = nThreads,' in draw,
        "model load passes selected thread count into SdCppEngine.create")
require('R.string.draw_taesd_enable' in activity and 'pickTaesdWeights()' in activity,
        "drawing settings expose visible TAESD opt-in and file selection")

# ModelLoader and sampler share sd_progress_cb in native code. Only sample-step-shaped
# progress values should be surfaced as denoising steps in the drawing UI.
require('@Volatile private var denoisingStarted = false' in draw,
        "denoising phase is tracked independently of model loading")
require('total in 1..(requestedSteps + 2)' in draw,
        "tensor-count progress is not mistaken for sampler progress")
require('if (denoisingStarted) c.getString(R.string.s_129) else c.getString(R.string.s_127)' in draw,
        "UI remains in model-loading phase until actual sampling starts")
require('private fun refreshTaesdUi()' in activity,
        "TAESD UI mirrors selected weights and enabled state")
require('nativeCreate(modelPath, vaePath, nThreads, wtype, flashAttn, backend, slots)' in engine,
        "Kotlin wrapper forwards the selected thread count to JNI")
require('p.n_threads    = (nThreads > 0) ? nThreads : sd_get_num_physical_cores();' in jni,
        "JNI forwards the requested count to sd.cpp")
require('int n_threads;' in header,
        "native context has an explicit thread-count parameter")

# Four LoRA slots: UI selection and per-slot weight flow through Kotlin, JNI, and sd.cpp.
require('private const val MAX_LORA_SLOTS = 4' in draw,
        "UI defines four LoRA slots")
require('private fun selectedLoras(page: String): List<SdCppEngine.Lora>' in draw,
        "generation resolves selected LoRA files and per-slot weights")
require('t2iLoraSpinners' in draw and 'i2iLoraSpinners' in draw,
        "text-to-image and image-to-image have independent LoRA selections")
require('data class Lora(val path: String, val scale: Float)' in engine,
        "Kotlin represents each LoRA as a path and multiplier pair")
require('loras.map { it.path }.toTypedArray()' in engine and 'loras.map { it.scale }.toFloatArray()' in engine,
        "Kotlin passes matched LoRA path and scale arrays to JNI")
require('jobjectArray loraPaths, jfloatArray loraScales' in jni,
        "JNI accepts multiple LoRA paths and scales")
require('g.lora_count = static_cast<uint32_t>(nativeLoras.size());' in jni,
        "JNI forwards all selected LoRAs to sd.cpp")
require('const sd_lora_t* loras;' in header and 'uint32_t lora_count;' in header,
        "sd.cpp generation parameters support multiple LoRAs")

# Exynos 2400e / Samsung Xclipse 940 Vulkan profile.
# Only the exact GPU name enables RDNA3 tuning; cooperative-matrix usage remains
# gated by the build-time shader support and the runtime driver's reported features.
require('static bool ggml_vk_is_xclipse_940(const vk::PhysicalDeviceProperties& props)' in vulkan
        and 'find("Xclipse 940")' in vulkan,
        "Vulkan backend detects the exact Samsung Xclipse 940 GPU")
require('return vk_device_architecture::AMD_RDNA3;' in vulkan
        and 'RDNA3 tile presets only for this exact GPU' in vulkan,
        "Xclipse 940 is mapped to the RDNA3 tuning family")
require('device->coopmat_support &&' in vulkan
        and 'ggml_vk_is_xclipse_940(device->properties)' in vulkan,
        "Xclipse RDNA3 matmul tuning is gated by runtime cooperative-matrix support")
require('Xclipse 940 profile detected' in vulkan
        and 'Keeping standard Vulkan matmul kernels' in vulkan,
        "Vulkan logs feature availability and preserves generic fallback")

# Use a newer host shader compiler only when it passes the exact cooperative-matrix shader probe.
require('PONKO_VULKAN_GLSLC' in cmake and 'using host glslc override' in cmake,
        "CMake can use an explicit host glslc while keeping the NDK fallback")
require('Probe host glslc cooperative-matrix support' in workflow
        and 'feature-tests/coopmat.comp' in workflow
        and 'PONKO_VULKAN_GLSLC=$GLSLC' in workflow,
        "CI probes host glslc against the cooperative-matrix shader before selecting it")
require('GGML_VULKAN_COOPMAT_GLSLC_SUPPORT' in workflow,
        "CI verifies the cooperative-matrix define reached the generated Android build")

# Runtime Vulkan diagnostics must be retrievable and visible on Android, not only stderr.
jni = read("app/src/main/cpp/sdcpp_jni.cpp")
require('ggml_log_set(ponkoGgmlLogCallback, nullptr)' in jni,
        "GGML logs are forwarded through Ponko's Android logger")
require('nativeVulkanDiagnostics' in jni,
        "JNI exposes the runtime Vulkan diagnostic report")
require('PonkoVulkan' in jni and 'matrix cores:' in jni,
        "Android Logcat captures Xclipse cooperative-matrix status")
require('target_include_directories(ponko_sd PRIVATE' in cmake and 'sd/ggml/include' in cmake,
        "JNI compiles against the bundled ggml logging API")
require('nativeVulkanDiagnostics' in engine,
        "Kotlin exposes Vulkan runtime diagnostics")
require('SdCppEngine.vulkanDiagnostics()' in draw,
        "model loading collects the current-device Vulkan capabilities")
require('activeVulkanDiagnostic.contains("cooperative matrix 1")' in draw
        and '" · Xclipse coopmat"' in draw and '" · Xclipse generic"' in draw,
        "model status uses the reported runtime feature flag to distinguish fast and fallback Vulkan paths")

print("All TAESD, CPU-thread, four-slot LoRA, and Xclipse Vulkan checks passed.")
