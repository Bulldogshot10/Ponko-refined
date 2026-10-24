# Native API parity checklist

Reference files:
- `app/src/main/java/com/litertchat/app/draw/SdCppEngine.kt`
- `app/src/main/cpp/sdcpp_jni.cpp`

This checklist is intentionally explicit. The Rust facade is not allowed to become the new default bridge until every row marked **Legacy only** has a Rust implementation and parity coverage. Current Rust exports are contract validation/introspection, not an inference replacement yet.

| Existing API / behavior | Rust facade state | Parity requirement before cutover |
|---|---|---|
| `nativeInfo` / sd.cpp version and system info | Legacy only | Same engine version and system-info output |
| `nativeVulkanDiagnostics` | Legacy only | Device count, Xclipse diagnostics, and logged diagnostics |
| `nativeLastParams` | Legacy only | Preserve complete last-context parameter dump and failure logs |
| `nativeListDevices` | Legacy only | Preserve device names/descriptions and empty result when unavailable |
| `nativeCreate` | Typed `ModelLoadRequest` contract + validator | Context creation/freeing; 13 ordered model slots; optional VAE; thread count; weight type including KEEP=-1; FlashAttention; direct VAE convolution; backend auto/CPU/Vulkan |
| `nativeGenerate` text-to-image | Typed `GenerationRequest` contract + validator | Prompt/negative prompt, dimensions, steps, CFG (including 0), seed, sampler, scheduler, ETA |
| `nativeGenerate` img2img | Typed `GenerationRequest` contract + validator | RGB888 bytes, source dimensions, strength normalization, correct buffer lifetime |
| HiRes path | Typed fields included | Enable flag, upscaler, model path, scale, target dimensions, steps, denoise strength, tile size |
| LoRA application | Typed arrays + count validation | Multiple LoRA paths and scales; do not introduce a new LoRA count cap |
| CLIP skip | Typed field included | Clamp to the current 1..12 behavior before native dispatch |
| Progress / preview callbacks | Legacy only | Thread attachment, Kotlin callback delivery, RGB preview bytes, exception handling and cleanup |
| `nativeCancel` | Legacy only | Cancellation reaches `sd_cancel_generation` and stays race-safe |
| `nativeFree` | Legacy only | No use-after-free while generation/API calls hold the handle |
| `nativeQuantizeUnet` | Typed request + quantization-type validator | q4_0/q5_0/q8_0 only; input/output checks; temporary GGUF; tensor/type/payload validation; atomic publish; never overwrite source/output |
| `nativeApplyResAdapter` | Typed request + validator | Optional VAE; normalization merge; same quantization rules; output validation; temporary publish |
| Single-flight generation / callback globals | Legacy behavior | Preserve serialization and reject conflicting work rather than racing handles |
| Bitmap conversion and output dimensions | Legacy behavior | Preserve RGB888/ARGB_8888 conversions and HiRes-returned width/height |
| CMake CPU profile and Vulkan fallback | Build configuration retained | Keep `-march=armv8.6-a+dotprod+i8mm+fp16` and the existing `-mtune=cortex-x4` profile enabled by default; Vulkan remains enabled by default |

## Cutover gate

1. Build and compile the opt-in Rust/UniFFI path with `-PponkoRustFacade=true`.
2. Add automated parity tests for validation, sampler/scheduler code mapping, model-slot mapping, quantization, and all request fields.
3. Exercise create/generate/cancel/free plus preview callbacks on the Exynos 2400e and confirm lifecycle safety.
4. Switch each Kotlin call site individually, preserving the old JNI entry point until the corresponding test passes.
5. Only then remove the matching JNI function. Never delete the whole bridge in one step.
