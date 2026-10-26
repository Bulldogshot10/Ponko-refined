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
- **VAE decode OOM retries — Partial, needs verification:** `app/src/main/cpp/sd/src/core/backend_fit.cpp` now progressively shrinks spatial tile dimensions over repeated decode retries, while retaining the existing temporal-tiling retry for supported video VAEs.

### Validation for this follow-up
- The memory-handling edits were committed to `feat/rust-uniffi-native-facade`; commits: `913f2eed` and `a9c416c8` (fusion, including merged CLIP QKV aliases), `ddd30e05`, `3be9625e`, and `eddc6201` (ADetailer), and `70191b90` (VAE retries).
- **Local validation not run:** no local checkout was available for running Rust/CMake/Gradle commands directly. Push-triggered CI on source commit `eddc6201` reports `Format Rust facade` and `Test Rust facade` completed successfully. The Rust-UniFFI APK build and the separate Vulkan/Exynos APK build were still in progress at the latest status check; no APK build result or target-device runtime test is available yet. Re-check the workflow results before treating native compilation as passed.
- **Next steps:** verify source-name matching and repeated parameter unregister/re-register with one and multiple SD1.x/SD2.x/SDXL LoRAs; run ADetailer with multiple detections at high resolution and compare merged/non-merged outputs; force VAE decode failures and confirm retries decrease tile size without altering successful decode output unexpectedly. Capture native logs/RSS and the actual failure signal to distinguish OOM from another fusion crash.


## CPU UFS tensor streaming experiment (2026-10-26)

### Partial implementation / needs verification
- **Experimental CPU streaming flag — implemented in source, needs build/runtime verification:** branch `feat/cpu-ufs-tensor-streaming` adds an opt-in UI preference, carries it through Kotlin → UniFFI → Rust → C ABI, forces CPU execution for this mode, and assigns diffusion weights to `params_backend=diffusion=disk`.
- **CPU segmented execution — implemented in source, needs verification:** when a graph has a valid multi-segment graph-cut plan and includes disk-resident weights, `GGMLRunner` may run it segmented even if a monolithic graph would fit. The disk-resident segment weights are unpinned and evicted after each segment.
- **Fallback behavior — implemented in source, needs verification:** if a streaming graph has no usable graph cuts, the runner logs a warning and executes monolithically; this path can still have a high memory peak.

### Current limitations
- CPU streaming is opt-in and applies on the next model load. It intentionally targets diffusion/UNet weights; text encoder, VAE, and other model components retain their normal residency in this first experiment.
- CPU async UFS prefetch is **not implemented**. Segment weights load synchronously from the model file, so repeated denoising steps may incur substantial storage I/O and slowdown.
- Page-cache reclamation is not forced. File-backed pages may remain in the kernel page cache after application buffers are released.
- No local checkout was available for a build or unit-test run during implementation; Android device/runtime validation on Exynos 2400e is still required.

### Next steps
1. Run `cargo fmt --manifest-path rust/ponko-native/Cargo.toml --all -- --check` and `cargo test --manifest-path rust/ponko-native/Cargo.toml`.
2. Build the ARM64 APK with `./gradlew :app:assembleDebug` and inspect C ABI / UniFFI generated binding errors.
3. On CPU, compare streaming off/on using the same SD1.5 GGUF, prompt, seed, resolution, steps, and thread count. Capture graph-cut logs, peak RSS, elapsed time, and storage-I/O behavior.
4. Verify with SDXL and other supported families whether graph-cut plans produce useful segments and whether the largest segment's working set remains bounded.
5. Only after measurements, experiment with bounded read-ahead and page-cache advice; do not label those as implemented until tested.
