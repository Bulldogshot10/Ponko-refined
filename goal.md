# Project Goal and Handoff

Branch: `feat/rust-uniffi-native-facade`. This inventory is specific to the UniFFI facade branch and must not be confused with the Hearth full-replacement branch.

## Goal
Move the Kotlin-to-native API into Rust through generated UniFFI bindings, with Rust owning request validation, callback routing, native handle management, and safe lifecycle coordination. The inference backend on this branch remains stable-diffusion.cpp + GGML behind a plain C ABI.

## Feature inventory

### Implemented / documented in this branch
- Rust crate at `rust/ponko-native/` and generated Kotlin-facing UniFFI API.
- Active Kotlin call path through the Rust facade instead of handwritten Java native-method declarations.
- Stable-diffusion.cpp inference retained behind `app/src/main/cpp/ponko_sd_c_api.h` / `.cpp`; the C++ inference implementation has **not** been replaced by Hearth on this branch.
- Context create/free, text-to-image, img2img, sampler/scheduler/ETA parameters, caller-provided LoRA paths/scales, HiRes settings, CLIP skip, progress/preview callbacks, and cancellation are included in the published API parity map.
- Vulkan diagnostics, last-parameter logs, device enumeration, UNet quantization, ResAdapter conversion, and LoRA fusion are also listed in `rust/ponko-native/API_PARITY.md`.
- Native context lifetime coordination prevents freeing a context while generation/cancel calls are using it; generation is single-flight due to global native callbacks.
- Android ARM64 Rust/UniFFI/C ABI/APK CI workflow is present.

### Partial / needs verification
- API parity is documented, but each native function and callback must pass Rust tests, Android build, and runtime testing before being declared stable.
- Review the current workflow triggers and inspect its latest run; do not infer success from workflow presence.
- ADetailer / YOLOv8 end-to-end status needs an explicit source-to-UI audit. The native API parity document does not list ADetailer as a facade API; a YOLOv8 header by itself does not prove that detection, cropping, parameter passing, and post-generation compositing are wired end to end.

### Not implemented on this branch's documented facade
- Full replacement of stable-diffusion.cpp drawing inference with Hearth. This branch intentionally retains the C++ inference backend; the separate `feat/hearth-native-rewrite` branch documents Hearth replacement.
- Any capability absent from `API_PARITY.md` must be treated as unverified until the active code, UI, and tests establish its status.

## Handoff notes for the next agent
1. Read `rust/ponko-native/README.md`, `rust/ponko-native/API_PARITY.md`, and `app/src/main/java/com/litertchat/app/draw/SdCppEngine.kt`.
2. Preserve API contract and persisted parameter behavior while changing the boundary; inspect Kotlin call sites before changing UniFFI signatures.
3. Keep context handles safe under concurrency and ensure callback-owned memory is copied before its native lifetime ends.
4. Run `cargo fmt --manifest-path rust/ponko-native/Cargo.toml --all -- --check`, `cargo test --manifest-path rust/ponko-native/Cargo.toml`, and `./gradlew :app:assembleDebug`.
5. Audit ADetailer as a complete feature: model loading, YOLOv8 inference, detection thresholds, crop/padding, face prompt/negative prompt, denoise/steps/CFG, post-processing, UI, errors/OOM handling, and real-device validation.
6. Keep this branch's backend scope distinct from `feat/hearth-native-rewrite`; don't mix the two architectures unless the user explicitly asks.

## Current blocker
Fresh build/CI and target-device behavior are not verified by this document. Do not report completion until the actual run results have been inspected.


## Memory handling follow-up (2026-10-26)

### Partial implementation / needs verification
- **LoRA fusion memory — Partial, needs verification:** `app/src/main/cpp/sd/src/convert.cpp` now retains LoRA source tensor names as metadata, registers only factors matching the current base tensor, and calls `release_loaded_tensors()` after each adapter application. This bounds retained LoRA parameter buffers instead of accumulating previously used tensors across the export. Expected tradeoff: more per-tensor adapter loading and I/O.
- **ADetailer memory — Partial, needs verification:** `app/src/main/cpp/sd/src/detailer.cpp` now creates and consumes non-merged full-resolution masks one at a time, merges incrementally in merged mode, and unloads the YOLOv8 detector/model manager after detection but before crop inpainting. The detector is lazily reloaded on the next ADetailer call, trading some load time for a lower inpainting memory peak.
- **VAE decode retry tiling — Restored to pre-regression behavior; device validation needed:** `app/src/main/cpp/sd/src/core/backend_fit.cpp` again performs only the original first temporal/spatial tiling fallback, then stops retrying if decode still fails. Progressive spatial tile shrinking was reverted because the user observed both slower generation and degraded VAE output.

### Validation for this follow-up
- The memory-handling edits were committed to `feat/rust-uniffi-native-facade`; commits: `913f2eed` and `a9c416c8` (fusion, including merged CLIP QKV aliases), `ddd30e05`, `3be9625e`, and `eddc6201` (ADetailer), and `70191b90` (VAE retries).
- **Prior native build validation:** push-triggered CI for source commit `eddc6201` completed successfully for both the Rust-UniFFI APK workflow and the Vulkan/Exynos APK workflow (runs [40270880986](https://github.com/Bulldogshot10/Ponko-refined/actions/runs/40270880986) and [40270881047](https://github.com/Bulldogshot10/Ponko-refined/actions/runs/40270881047)). Those builds predate the CPU-only/VAE-revert change and are not validation for the current source. No local checkout or target-device test is available.
- **Next steps:** verify source-name matching and repeated parameter unregister/re-register with one and multiple SD1.x/SD2.x/SDXL LoRAs; run ADetailer with multiple detections at high resolution and compare merged/non-merged outputs; force VAE decode failures and confirm retries decrease tile size without altering successful decode output unexpectedly. Capture native logs/RSS and the actual failure signal to distinguish OOM from another fusion crash.


## Performance regression follow-up (2026-10-26)

### Requested changes applied; needs build/device verification
- **Vulkan — Disabled by default:** `app/build.gradle.kts` and `app/src/main/cpp/CMakeLists.txt` default Vulkan to OFF; both APK workflows explicitly pass `-PponkoEnableVulkan=OFF`, and `rust/ponko-native/build.rs` defaults the Vulkan build environment to OFF. The Exynos 2400E CPU ISA profile remains ON. This is a CPU-only stable-diffusion.cpp build configuration, not a claim that CPU will be faster than a functioning Vulkan backend.
- **VAE retry behavior — Reverted:** `prepare_vae_decode_retry_tiling()` restores the former behavior: enable the first temporal/spatial tiling fallback, and return false when that fallback has already been attempted. The progressive smaller-tile retry logic was removed to restore previous decode behavior/output characteristics.
- **Observed performance regression (user-reported):** 6-step generation previously took about 120 seconds, and now takes 200+ seconds without ADetailer and 300+ seconds with ADetailer. The ADetailer-off slowdown means ADetailer is not the root cause of the baseline regression; ADetailer/inpainting adds further runtime. Root cause of the baseline slowdown remains unverified.
- **Validation still pending:** no new CPU-only APK build or real-device comparison has been run yet. Previous successful APK runs used the Vulkan-enabled configuration and do not validate this change. Do not infer improved speed or restored image quality until tested on-device.
- **Next steps:** build the CPU-only APK and verify the packaged native library has no Vulkan backend; compare the same model, dimensions, sampler, CFG, 6 steps and fixed seed against the earlier ~120-second baseline; record step timing and VAE logs. Then measure ADetailer separately to distinguish detector load, detection, crop inpainting and compositing costs.


## ADetailer pre-crop latency follow-up (2026-10-26)

### Implemented; performance still needs device verification
- `app/src/main/cpp/sd/src/detailer.cpp`: optimized full-frame RGB copying. When the generated source is already RGB at the target dimensions, ADetailer now copies bytes directly instead of running every pixel through a bilinear sampler that cannot change the dimensions or pixel values. Other channel counts are converted to RGB directly without resampling.
- `morphology_mask()`: compute dilation/erosion only in the region where the result can be nonzero, while preserving the same mask dimensions and kernel semantics. It still scans the input once to determine the occupied region; the expensive per-pixel kernel is no longer run across the whole image for every detection.
- Added stage timings for detector input preprocessing, YOLO inference, detection decoding/NMS, detector (re)load and unload, mask preparation, full-frame RGB copy, crop preparation, crop inpainting, and compositing. `app/src/main/cpp/ponko_sd_c_api.cpp` now logs base-generation time and detector load time separately. Use Logcat output tagged `PonkoSdRust` to identify remaining delay before the first crop.
- **Not yet verified:** no native build or device benchmark has been run for these changes. The intended performance improvements have not been measured; compare before/after with identical detector, output size, ADetailer args, base steps and seed. CI is skipped on this change to avoid consuming the repository's GitHub Actions allowance.
- **Next step:** build locally and capture the `ADetailer timing` and `BENCHMARK` lines. If detector reload/inference dominates, reduce detector `input_size` via `extra_ad_args` (e.g. compare 640 vs 384) and validate missed small faces/hands; if mask prep dominates, inspect detection counts and whether merge mode is enabled.


## Model conditioning feature follow-up (2026-10-26)

### Implemented in source; CI build passed; device verification pending
- **ControlNet path:** added the optional `control_net` model-set slot and routes an RGB888 control map plus strength through Kotlin, UniFFI, Rust's C ABI mirror and `ponko_sd_generate()` into `sd_img_gen_params_t.control_image/control_strength`. The image should be preprocessed for the selected ControlNet. A full-frame map is not reused for ADetailer crop passes because it would be spatially misaligned.
- **IP-Adapter path:** added the optional `ip_adapter` model-set slot, reference-image picker, strength control and native `ip_adapter_image/ip_adapter_strength` wiring. Use weights compatible with the active SD model and a matching CLIP Vision component where required.
- **Textual Inversion:** added one optional `embedding` slot. For SD1.x/SD2.x/SDXL, the bridge passes an `sd_embedding_t` to the existing native loader; the prompt trigger token defaults to the embedding filename stem (without extension).
- **UI scope:** conditioning image controls are shared by text-to-image and img2img. Image selections remain in memory for the current page session; strength settings persist in preferences. Conditioning weights are optional model-set slots.
- **Validation in code:** Rust validates optional image dimensions and RGB888 length, plus finite strengths; C++ repeats image-buffer validation at the ABI boundary and rejects images when the corresponding ControlNet/IP-Adapter weights are not loaded.
- **Static audit follow-up (2026-10-26):** bumped the UniFFI facade contract to version 4 for the changed typed generation request; updated parity documentation from 13 to 16 model slots and corrected its Vulkan default statement; added a Rust unit test for the native slot ordering; added Chinese translations for the conditioning UI.
- **Build validation:** CI build succeeded on 2026-10-26 in [run 40310371991](https://github.com/Bulldogshot10/Ponko-refined/actions/runs/40310371991), building source commit `566b39d3faef8ba134016064960af61839252517`. Rust formatting/tests, Gradle APK build, required-library verification, and artifact upload all passed.
- **Needs device verification:** compiled wiring has not yet been exercised on-device with actual compatible weights. Test a compatible SD1.5 ControlNet map, IP-Adapter weights with CLIP Vision, and the embedding trigger token.

## Conditioning upload UI correction (2026-10-26)

- **Bug confirmed and fixed:** the Conditioning panel previously showed image pickers for ControlNet and IP-Adapter but no explicit model-weight import buttons in that panel. The ControlNet card's only pick button opened the image picker, confusing model weights with the condition-map image.
- **Implemented:** added distinct import/replace controls for ControlNet weights (`control_net`), IP-Adapter weights (`ip_adapter`), CLIP Vision weights (`clip_vision`), and Textual Inversion embedding (`embedding`). These route into the existing selected-model-set slot importer. ControlNet map and IP-Adapter reference image pickers remain separate image-only actions.
- **UI and validation:** the panel now displays selected weight filenames and reminds users to reload the drawing model after changing weights. Import is blocked until a model set is selected; slots not supported by the current model family are rejected.
- **Build validation succeeded on the exact source commit** `cea024c1ae1b709abe58233a4f40d242256bde26`: [CPU/Exynos APK build #40315970274](https://github.com/Bulldogshot10/Ponko-refined/actions/runs/40315970274) and [Rust-UniFFI APK build #40315970261](https://github.com/Bulldogshot10/Ponko-refined/actions/runs/40315970261) both passed. Rust formatting/tests, Gradle APK build, UniFFI Kotlin binding generation, and packaged native-library verification all succeeded.
- **Fresh CPU APK:** artifact `ponko-cpu-exynos2400e-debug-apk-40315970274`; APK SHA-256 `62b3fc322b98914b4eb0bb979fd8a04cf34c0cc9b86a96a07dc1900863e89a78`. The downloaded APK hash matches the CI-produced checksum.
- **Still pending:** there has not yet been an install/device test with real ControlNet and IP-Adapter weights. Build success proves compilation and packaging, not successful image generation on-device.
- **Next step:** install the freshly built APK on the target phone, confirm the four model import buttons open the model-slot picker, verify the two image buttons open the image picker, then test actual compatible ControlNet/IP-Adapter weights and trigger embedding.

## CPU APK build follow-up (2026-10-26)

- **Build trigger enabled:** `.github/workflows/ponko-app-apk.yml` now also runs when `goal.md` changes, so documented build/validation handoffs can request a fresh APK build without touching application source. The workflow-file change triggered this build against the conditioning-enabled branch head.
- **Build result: SUCCESS.** [GitHub Actions run 40310371991](https://github.com/Bulldogshot10/Ponko-refined/actions/runs/40310371991) completed successfully on source commit `566b39d3faef8ba134016064960af61839252517`. `cargo fmt`, `cargo test`, `./gradlew :app:assembleDebug`, required native-library checks, and upload all passed. Gradle reported `BUILD SUCCESSFUL in 1m 17s`.
- **APK:** artifact `ponko-cpu-exynos2400e-debug-apk-40310371991`, APK size 51,697,848 bytes. SHA-256: `685429588ae9ffb806cedcae1239dbd0ee4066304910d1e224f1e29185966333`. Downloaded artifact checksum matches exactly.
- **Native libraries verified:** `lib/arm64-v8a/libomp.so`, `libponko_native.so`, `libponko_sd.so`, and `libstable-diffusion.so` are present. Build configuration is Vulkan OFF with the Exynos 2400E CPU profile ON.
- **Validation boundary:** this confirms CI compilation and packaged-library verification. No install or on-device generation test has been performed. ControlNet, IP-Adapter, and Textual Inversion still need runtime tests with compatible model weights.
- **Next step:** install the APK on the target device, confirm app startup, then test ControlNet, IP-Adapter with its CLIP Vision dependency as applicable, and the embedding trigger token.

## Conditioning model-load failure audit (2026-10-26)

### Implemented in source; fresh build and device validation pending
- **ControlNet format conversion:** added Diffusers-style ControlNet tensor-name conversion for SD1.x/SD2.x, including the condition embedding, down-block zero convolutions, and middle-block output. Previously this conversion ran only for SDXL, so Diffusers-format Canny/OpenPose weights for SD1.x/SD2.x could remain under incompatible names. Original LDM/lllyasviel-style `control_model.*` weights keep their existing path.
- **Native failure diagnostics:** collect distinct native GGML/stable-diffusion.cpp error messages (up to 8 KiB) during context creation and propagate them through Rust/UniFFI into Kotlin. The drawing page records the captured detail in `.loadstage`, returns a clipped excerpt with the load failure, and keeps the full log file instead of deleting it on failure.
- **IP-Adapter status:** source wiring and tensor-name mapping were audited, but there is not yet enough evidence to claim its runtime load failure is fixed. The propagated native error should identify the exact missing/mismatched tensor or backend failure on the next attempt. Do not mark IP-Adapter runtime loading as implemented until a compatible SD1.5 or SDXL adapter plus matching CLIP Vision weights load on-device.
- **Validation still pending:** `cargo fmt --manifest-path rust/ponko-native/Cargo.toml --all -- --check`, `cargo test --manifest-path rust/ponko-native/Cargo.toml`, `./gradlew :app:assembleDebug`, plus device tests using compatible Canny/OpenPose ControlNet and IP-Adapter weights. Compilation is not proof of correct tensor compatibility or successful on-device generation.


## Two-pass conditioning pipeline (2026-10-26)

- **Implemented in source; build and device verification pending:** optional UI modes `Off`, `ControlNet draft → base-model refine`, and `IP-Adapter draft → base-model refine`, available to both txt2img and img2img. Chat/API calls keep single-pass behavior unless explicitly opted in through the function argument.
- **Pass 1:** reload a context containing only the selected conditioning family (ControlNet alone, or IP-Adapter plus CLIP Vision), then render a short draft with configurable 1–12 steps (default 4).
- **Pass 2:** free the conditioning context and reload the same model set without ControlNet, IP-Adapter, or CLIP Vision. Feed the draft through img2img for refinement with configurable denoise strength (default 0.45); use the normal generation step count, LoRAs, HiRes and ADetailer settings for this pass.
- **Memory/lifecycle behavior:** swap contexts sequentially after freeing the prior handle; the base-only context remains active after the two-pass run. If the user switches back to single-pass conditioning and a conditioning image is selected, the full model-set context is restored before rendering. The model-set files themselves are not modified.
- **CI build validation: PASSED on commit `fe01c63f2a41995bbb0c4f8f12e52366ac9c819a` (2026-10-26).** CPU/Exynos APK run [40332602029](https://github.com/Bulldogshot10/Ponko-refined/actions/runs/40332602029) succeeded, including Rust format/tests, Gradle APK build, packaged-library verification, and artifact upload. Rust-UniFFI APK run [40332602056](https://github.com/Bulldogshot10/Ponko-refined/actions/runs/40332602056) also succeeded, including Rust format/tests, APK build, generated UniFFI binding inspection, packaged-library verification, and artifact upload. CPU artifact `ponko-cpu-exynos2400e-debug-apk-40332602029` has APK SHA-256 `3bed20d84b0f92787af384fbf7a4ce17b5cb8a66f63d2585fce19639ed268c26`.
- **Device/runtime validation still required:** test two-pass generation in both txt2img and img2img with SD1.5 + compatible Canny/OpenPose weights and with IP-Adapter + matching CLIP Vision weights. Check cancellation between passes, verify the base context remains usable, and measure time/RAM. Two passes reduce simultaneous conditioning residency; they do not guarantee lower total wall-clock time.

## PyTorch .bin → safetensors conversion (2026-10-27)

- **Source change merged to the requested branch:** commit `6fb2ae6cc11c657a439d37061eddd590c032cf3c` adds the drawing-tool action, native converter, C ABI, Rust/UniFFI API v5 and Kotlin wrapper. The user selects a PyTorch `.bin` and destination with Android document pickers.
- **Data/output behavior:** retains source tensor names and stored dtypes (no name remapping or quantization), validates the output, stages then atomically publishes it, and leaves the original checkpoint untouched.
- **CI passed:** CPU/Exynos APK workflow run [40470498458](https://github.com/Bulldogshot10/Ponko-refined/actions/runs/40470498458) and Rust-UniFFI APK workflow run [40470498309](https://github.com/Bulldogshot10/Ponko-refined/actions/runs/40470498309). Both passed Rust formatting/tests, Android APK build, packaged native-library checks, and artifact upload; the UniFFI run also inspected the generated Kotlin binding for `convertTorchBinToSafetensors`.
- **Artifacts:** `ponko-cpu-exynos2400e-debug-apk-40470498458` and `ponko-rust-uniffi-build-40470498309`. CPU/Exynos APK SHA-256: `d9eab0f991d3f8b6ee7e6576de3f5b20f58e8eee69ed40618a0d18f2ab99e21e`.
- **Compatibility boundary / remaining test:** conversion changes the file container only. It does not implement FaceID-specific IP-Adapter inference. The actual target `.bin` still needs to be converted on-device and tested with the backend; CI verifies build integration, not the runtime behavior of a particular checkpoint.


## PyTorch conversion and fusion crash audit (2026-10-28)

### Current status: Partial; device/runtime validation required
- The `.bin` button is wired end-to-end through Kotlin -> generated UniFFI -> Rust -> the C ABI -> stable-diffusion.cpp's streaming exporter. This proves the call path exists, not that every PyTorch checkpoint variant is supported.
- Input coverage is restricted to state-dict-like tensor layouts parsed by the bundled pickle reader (protocols 2–5) in supported PyTorch ZIP or legacy containers. Whole `nn.Module`/arbitrary Python object graphs, unsupported pickle layouts/opcodes, and legacy tar checkpoints are not supported as a general-purpose conversion format. `ModelLoader` filters known unused tensors, so this is not guaranteed to preserve every source key, despite preserving names/dtypes for tensors that are exported.
- Crash-related fixes pushed to this branch: commit `f72658486ab069eb460e36ac32151a576d229b88` bounds legacy-checkpoint metadata reads to a 64 MiB prefix instead of copying the entire multi-GB file into RAM; commit `b5fd6304113458d82bdca94e6da048aa376e77e6` fixes the stream memory-budget reservation so a single oversized tensor cannot bypass the limit and estimates source + destination + serialized output memory.
- The earlier single-writer change (`75a39ea93060189fbf6340bd2f73b9bb38a1ac7e`) alone did not stop the user's crash. The two new memory fixes are source-level mitigations; they are **not yet proven to fix the observed device crash** and may reject single tensors whose estimated working set exceeds the configured budget.
- **Build/test status:** source-level audit completed; the current post-patch APK CI run and real-device result must be checked separately. Do not mark as fully supported until valid checkpoint fixtures for modern ZIP and legacy format pass conversion and the resulting files are loaded/tested on the target device. Capture Logcat tagged `PonkoSdRust` and Android low-memory-killer evidence for any repeated crash.
- **Next steps:** inspect post-patch CPU/Exynos and Rust-UniFFI APK workflow runs; test a small supported state_dict fixture plus realistic SD1.5/SDXL checkpoints in ZIP format, and a supported legacy file if available; verify key/tensor count and dtype/shape against expectations; test unsupported whole-module/tar inputs fail cleanly; then reproduce on-device with Logcat and memory/RSS logs.


## stable-diffusion.cpp upstream-port CI repair (2026-10-29)

- **Target branch:** \`feat/sdcpp-ggml-upstream-port\`.
- **Latest observed failing run before this repair:** [run 40843412900](https://github.com/Bulldogshot10/Ponko-refined/actions/runs/40843412900). Rust formatting and all 13 Rust unit tests passed; the Android native build failed compiling the overlay-added \`src/tokenizers/gpt_oss_tokenizer.cpp\`.
- **Root cause:** the pinned stable-diffusion.cpp revision \`77b72a2efed78f55de26892c7e909cee001e8b83\` declares \`BPETokenizer(const std::string& pattern)\` with no default constructor. The overlay-defined \`GPTOSSTokenizer\` constructor did not initialize that base class, so Clang rejected the C++ translation unit.
- **Fix committed in overlay patch:** initialize \`BPETokenizer\` with the Unicode-aware \`o200k_base\` token-splitting pattern, and update the added-file hunk length. This corrects the API mismatch and preserves the GPT-OSS tokenizer's intended token boundaries.
- **Earlier CI failures traced:** run [40842595609](https://github.com/Bulldogshot10/Ponko-refined/actions/runs/40842595609) failed while applying the converter overlay because of upstream line endings; the next revision added line-ending normalization and advanced to C++ compilation. Run [40839420956](https://github.com/Bulldogshot10/Ponko-refined/actions/runs/40839420956) had an earlier \`backend_fit.cpp\` patch-context failure; later overlay revisions advanced beyond configuration and compiled that source.
- **Validation status:** fix pushed to the requested branch. Fresh CI build result must be inspected before marking the APK build successful; Rust tests passing alone do not establish APK success.
- **Next step:** inspect the new \`ponko-app-apk.yml\` run. If native compilation advances to another error, use that run's exact compiler output, repair the overlay against the pinned upstream source, and repeat until APK packaging and required-library verification complete.
