# Native API parity and migration map

Reference files:
- `app/src/main/java/com/litertchat/app/draw/SdCppEngine.kt`
- `app/src/main/cpp/ponko_sd_c_api.h`
- `app/src/main/cpp/ponko_sd_c_api.cpp`
- `rust/ponko-native/src/lib.rs`

The Kotlin-facing JNI declarations have been removed from the active build. Kotlin now calls generated UniFFI functions and the `NativeEngine` object. The C++ inference engine remains intact behind a plain C ABI.

| Previous API / behavior | New path | Preserved behavior |
|---|---|---|
| `nativeInfo` | `NativeEngine.info()` -> `ponko_sd_info` | sd.cpp version and system info |
| `nativeVulkanDiagnostics` | `NativeEngine.vulkanDiagnostics()` -> `ponko_sd_vulkan_diagnostics` | Vulkan device status; CPU-only builds report that Vulkan was not compiled |
| `nativeLastParams` | `NativeEngine.lastParams()` -> `ponko_sd_last_params` | Complete last-context parameter dump |
| `nativeListDevices` | `NativeEngine.listDevices()` -> `ponko_sd_list_devices` | Existing tab-separated device list |
| `nativeCreate` | `ModelLoadRequest` -> `NativeEngine.createContext` -> `ponko_sd_create` | 16 model slots, including ControlNet, IP-Adapter, and one Textual Inversion embedding, plus optional VAE, thread count, KEEP=-1, FlashAttention, direct VAE convolution, and backend selection |
| Text-to-image | `GenerationRequest` -> `generateImage` -> `ponko_sd_generate` | Prompt, negative prompt, dimensions, steps, CFG, seed, sampler, scheduler, ETA |
| Img2img | `GenerationRequest.initImageRgb888` -> `ponko_sd_generate` | RGB888 image bytes, dimensions, normalized strength |
| HiRes | Typed fields in `GenerationRequest` | Upscaler, model path, scale, dimensions, steps, denoise strength, tile size |
| LoRAs | `loraPaths` + `loraScales` | All matched path/scale entries are preserved without an arbitrary count cap |
| ControlNet | `ModelSlots.controlNet` + `GenerationRequest.controlImageRgb888` | Loads optional ControlNet weights and forwards an RGB888 map plus strength; use a preprocessed map compatible with the selected ControlNet |
| IP-Adapter | `ModelSlots.ipAdapter` + `GenerationRequest.ipAdapterImageRgb888` | Loads optional adapter weights and forwards an RGB888 reference image plus strength; compatible CLIP Vision weights may be required in `clipVision` |
| Textual Inversion | `ModelSlots.embedding` -> `sd_ctx_params.embeddings` | Loads one optional embedding for SD1.x/SD2.x/SDXL; trigger token defaults to the filename stem |
| CLIP skip | `GenerationRequest.clipSkip` | Clamped to 1..12 in the C++ adapter |
| Progress callback | UniFFI `StepCallback` -> Rust trampoline -> C ABI callback | Synchronous callback during native generation |
| Preview callback | UniFFI `PreviewCallback` -> Rust trampoline -> C ABI callback | RGB888 previews; temporary native memory is copied before callback return |
| `nativeCancel` | `NativeEngine.cancel` -> `ponko_sd_cancel` | Calls `sd_cancel_generation(..., SD_CANCEL_ALL)` |
| `nativeFree` | `NativeEngine.freeContext` -> `ponko_sd_free` | Rust read/write locking prevents freeing a context during active native calls |
| LoRA fusion | `LoraFusionRequest` -> `NativeEngine.fuseLoras` -> `ponko_sd_fuse_loras` -> `fuse_loras_to_file` | Multi-LoRA weighted fusion to a new GGUF; F16 or Q4/Q5/Q8 UNet output; streamed export, original inputs preserved |
| `nativeQuantizeUnet` | `QuantizeUnetRequest` -> `ponko_sd_quantize_unet` | q4_0/q5_0/q8_0, source checks, temporary GGUF, tensor/type/payload validation, atomic publish |
| `nativeApplyResAdapter` | `ResAdapterRequest` -> `ponko_sd_apply_resadapter` | Optional VAE, normalization merge, same quantization rules and output validation |
| PyTorch `.bin` conversion | `NativeEngine.convertTorchBinToSafetensors` -> `ponko_sd_convert_torch_bin_to_safetensors` -> stable-diffusion.cpp `ModelLoader` / streaming writer | **Partial, restricted input support:** handles compatible PyTorch tensor `state_dict` layouts in supported ZIP or legacy containers and validates the emitted safetensors structure. It is not a general-purpose PyTorch serializer: arbitrary pickled modules/checkpoint objects are unsupported, and known unused tensors are filtered by the model loader, so exact one-to-one tensor preservation is not guaranteed. Source is left untouched. Runtime/device validation required. |
| Single-flight generation | C++ generation mutex + global callback reset | Conflicting generation attempts fail as busy instead of racing global callback state |
| Bitmap conversion / output size | Rust RGB888 result -> Kotlin's ARGB result contract | Existing `DrawPage` and image API call sites keep the same interface |
| CPU profile / Vulkan | CMake profile and backend options | Keeps `-march=armv8.6-a+dotprod+i8mm+fp16 -mtune=cortex-x4` by default; Vulkan is disabled by the current build default unless explicitly enabled |

## Verification still required

1. Run `cargo fmt --check` and `cargo test`.
2. Build the Android APK with Rust stable, `aarch64-linux-android`, cargo-ndk, Android NDK 27.3, and CMake 3.31.6 installed.
3. On the Exynos 2400e, test model creation/free, txt2img, img2img, multiple LoRAs, HiRes, progress/preview callback, cancellation, device diagnostics, UNet quantization, and ResAdapter.
4. Compare representative outputs and timings against the previous build before declaring runtime parity complete.
5. On compatible SD1.5/SDXL model sets, test ControlNet with a matching preprocessed map, IP-Adapter with matching weights and CLIP Vision, and Textual Inversion using the embedding filename stem as the prompt token. Build and device verification are still required.

The old `sdcpp_jni.cpp` source is no longer part of the build. The Kotlin-facing JNI interface is replaced; stable-diffusion.cpp and GGML remain C++ by design. The UniFFI facade contract version is 5.
