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

## CPU APK build follow-up (2026-10-26)

- **Build trigger enabled:** `.github/workflows/ponko-app-apk.yml` now also runs when `goal.md` changes, so documented build/validation handoffs can request a fresh APK build without touching application source. The workflow-file change triggered this build against the conditioning-enabled branch head.
- **Build result: SUCCESS.** [GitHub Actions run 40310371991](https://github.com/Bulldogshot10/Ponko-refined/actions/runs/40310371991) completed successfully on source commit `566b39d3faef8ba134016064960af61839252517`. `cargo fmt`, `cargo test`, `./gradlew :app:assembleDebug`, required native-library checks, and upload all passed. Gradle reported `BUILD SUCCESSFUL in 1m 17s`.
- **APK:** artifact `ponko-cpu-exynos2400e-debug-apk-40310371991`, APK size 51,697,848 bytes. SHA-256: `685429588ae9ffb806cedcae1239dbd0ee4066304910d1e224f1e29185966333`. Downloaded artifact checksum matches exactly.
- **Native libraries verified:** `lib/arm64-v8a/libomp.so`, `libponko_native.so`, `libponko_sd.so`, and `libstable-diffusion.so` are present. Build configuration is Vulkan OFF with the Exynos 2400E CPU profile ON.
- **Validation boundary:** this confirms CI compilation and packaged-library verification. No install or on-device generation test has been performed. ControlNet, IP-Adapter, and Textual Inversion still need runtime tests with compatible model weights.
- **Next step:** install the APK on the target device, confirm app startup, then test ControlNet, IP-Adapter with its CLIP Vision dependency as applicable, and the embedding trigger token.
