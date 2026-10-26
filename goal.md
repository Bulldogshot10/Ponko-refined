# Project Goal and Handoff

Branch: `feat/hearth-native-rewrite`. This file is branch-specific. Do not apply its Hearth-only claims to `main` or another branch.

## Goal
Replace the drawing inference backend completely with Rust Hearth + Burn/WGPU Vulkan while preserving unrelated Ponko app capabilities. This branch intentionally has **one drawing backend**, not a parallel stable-diffusion.cpp fallback.

## Feature inventory

### Implemented / present in branch docs and source
- Drawing adapter in Rust at `rust/ponko-native/` with Kotlin-facing API in `app/src/main/java/com/litertchat/app/draw/HearthEngine.kt`.
- Hearth/Burn/WGPU Vulkan as the only drawing execution backend; no stable-diffusion.cpp drawing backend or C++ diffusion bridge is intended in this branch.
- SafeTensors checkpoints for SD 1.5 and SDXL.
- LoRA application to supported CLIP/UNet layers with per-adapter strength.
- Documented sampler subset: Euler, Euler a, DPM++ 2M, DPM++ SDE; Normal and Karras schedules.
- Progress callbacks and cancellation between denoising steps.
- Android ARM64 Rust/UniFFI/APK workflow configured in `.github/workflows/rust-uniffi-android.yml`.

### Not implemented in the documented Hearth adapter
- Drawing GGUF/quantized model loading.
- Img2img, HiRes Fix, ADetailer, UNet quantization, ResAdapter conversion, and LoRA fusion.
- Broader model families beyond SD 1.5 and SDXL.

### Partial / needs verification
- Exynos 2400e / Xclipse 940 Vulkan execution on a physical device is explicitly pending in `rust/ponko-native/API_PARITY.md`.
- SD 1.5 and SDXL model-loading and generation checks, memory behavior, Vulkan diagnostics, image dimensions, and cancellation still need to be confirmed against an installed APK on the target device.
- Build success, runtime success, and feature parity are separate statuses. Don't mark device validation as complete after CI alone.

## Handoff notes for the next agent
1. Read `rust/ponko-native/README.md` and `rust/ponko-native/API_PARITY.md` first.
2. Preserve the full backend replacement. Do not restore a second C++/stable-diffusion.cpp drawing backend or CPU fallback unless the user explicitly changes the architecture.
3. Recursively fetch Hearth/Burn/CubeCL/Cubek submodules; check `.gitmodules` when the build cannot resolve native source.
4. Run `cargo fmt --manifest-path rust/ponko-native/Cargo.toml --all -- --check`, `cargo test --manifest-path rust/ponko-native/Cargo.toml`, and `./gradlew :app:assembleDebug` (or `gradlew.bat` on Windows).
5. For device validation, record model format, checkpoint/VAE, prompt, resolution, sampler/schedule, memory usage, Vulkan logs, cancellation result, phone model, OS build, APK SHA, and commit.
6. Update this document and `API_PARITY.md` together when support changes.

## Current blocker
Real-device Xclipse 940 validation is pending. No fresh CI/build/device result is claimed here.
## Peer review audit (2026-10-26)

### Changes made on `feat/hearth-native-rewrite`

- Hardened Rust generation request validation: dimensions are limited to 64–2048 pixels per edge and must be divisible by 8; no more than four LoRAs are accepted. Added regression assertions for oversize dimensions, excessive LoRAs, and case-insensitive SDXL family matching.
- Added checked multiplication for the RGB output byte count, preventing integer overflow before the output buffer allocation.
- Moved Burn/WGPU initialization until after checkpoint, tokenizer, and VAE path checks so invalid local inputs return useful validation errors before Vulkan startup.
- Prevented API generation from entering the same Hearth handle while the UI/chat generation flag is active.
- Guarded model changes and native unloads while generation is active. Model replacement detaches `DrawCore` before freeing the previous native handle, so a failed replacement cannot leave the API referencing a freed handle.
- Hid the result-page img2img action and removed img2img from the chat image menu because the current Hearth adapter is text-to-image only. Added localized user feedback if an old caller still invokes the img2img entry point.
- Fixed the Android validation workflow's pull-request path filter so changes under `scripts/**`, including the CubeCL Android compatibility patch, are included in CI.
- Updated both README languages and `API_PARITY.md` to match the actual Hearth feature boundary and request limits.

### Validation status

- Source-level audit and patch-anchor checks completed through the GitHub repository API.
- `cargo fmt`, `cargo test`, and `./gradlew :app:assembleDebug` were **not run** in this environment: a local checkout failed because the environment could not resolve `github.com`. No build or test pass is claimed.
- GitHub Actions was not manually triggered. Before device testing, run `cargo fmt --manifest-path rust/ponko-native/Cargo.toml --all -- --check`, `cargo test --manifest-path rust/ponko-native/Cargo.toml`, and `./gradlew --no-daemon :app:assembleDebug` in a recursive-submodule checkout.
- Still required: a real SD 1.5 SafeTensors text-to-image smoke test and a memory/cancellation check on Exynos 2400e / Xclipse 940; SDXL needs its own load/generation test where device memory allows.

### Remaining Hearth limitations

- Drawing GGUF/quantization, img2img, HiRes Fix, ADetailer, UNet quantization, ResAdapter conversion, and LoRA fusion remain unsupported and must stay out of the active drawing UI.
- The repository's current device-validation status remains pending until real-device logs are available.


## Follow-up peer review (2026-10-26)

### Additional changes on `feat/hearth-native-rewrite`

- Wired CLIP Skip all the way from the Kotlin UI through the UniFFI request into Hearth. SD 1.5 now keeps the normal final-layer output at skip 1 and selects earlier hidden layers for higher values; SDXL preserves its existing penultimate CLIP-L output at skip 1 and selects earlier CLIP-L layers for higher values.
- Added Rust validation for CLIP Skip values outside 1–12 and unit coverage for the SD 1.5 / SDXL layer mapping.
- Removed the visible ETA control because the current Hearth sampler calls do not consume the legacy ETA parameter. The legacy field remains in the API structure for compatibility.
- Replaced silent truncation of mismatched LoRA path/weight arrays with explicit Kotlin validation; the native request still enforces the four-adapter maximum.
- Hardened native lifecycle serialization: overlapping generation calls now fail immediately instead of queuing, and native context release refuses to run while generation holds the shared WGPU runtime.
- Added post-decode synchronization and Burn/WGPU memory cleanup so successful and failed RGB conversion paths release cached VAE decode allocations after each generation.
- Direct callers now receive explicit failures for unsupported img2img, HiRes Fix, and ADetailer options rather than silently getting a text-to-image result.
- Updated `rust/ponko-native/API_PARITY.md` with CLIP Skip and ETA status.

### Validation status for this follow-up

- Source/patch-anchor verification completed through the GitHub repository API; changed files were fetched again after writing.
- `git ls-remote` / local checkout failed with `Could not resolve host: github.com`. Therefore `cargo fmt --check`, `cargo test`, and `./gradlew --no-daemon :app:assembleDebug` have **not** been run here.
- The available commit-status query returned no status checks for the latest code commit. No automated build or test pass is claimed.
- Next: in a recursive-submodule checkout, run `cargo fmt --manifest-path rust/ponko-native/Cargo.toml --all -- --check`, `cargo test --manifest-path rust/ponko-native/Cargo.toml`, and `./gradlew --no-daemon :app:assembleDebug`. Then install the APK and test SD 1.5 generation, CLIP Skip 1 vs 2, cancellation while model weights load, repeated generations for memory growth, and SDXL VAE decoding on Exynos 2400e / Xclipse 940.


### Cancellation-race follow-up (2026-10-26)

- Added a UniFFI `reset_cancel(handle_id)` method and call it immediately before Kotlin enters the blocking `generate_image` call. Removed the second reset from Rust generation startup, preventing a cancel request arriving during native-call startup from being cleared.
- Validation remains source-only because local GitHub DNS resolution prevents checkout. Re-run Rust formatting/tests and the Android debug build using the commands above; test cancel immediately after the UI button changes to “stop” and verify a subsequent generation still starts normally.


### Memory and CLIP conditioning follow-up (2026-10-26)

- Added final LayerNorm after selecting an earlier SD 1.5 CLIP hidden state for CLIP Skip; standard CLIP conditioning normalizes skipped hidden states too.
- Added Burn/WGPU memory cleanup between CLIP and UNet loading, and a scope guard that syncs/cleans cached backend allocations when either SD 1.5 or SDXL render exits early with an error.
- Updated `API_PARITY.md` to clarify the SD 1.5 CLIP Skip normalization behavior.
- These changes are source-reviewed but **not compiled or tested** here due the checkout/DNS blocker. The next validation run must confirm Rust 0.20 API compatibility for `LayerNorm` construction and exercise repeated SD 1.5/SDXL generations plus cancellation/error paths.


### Final audit pass (2026-10-26)

- Hardened cancellation reset: the new `reset_cancel` method now refuses to clear a cancellation flag while a generation holds the native generation lock. Kotlin calls it immediately before the blocking UniFFI generation call; Rust no longer resets the flag after the request may already have been cancelled.
- Added memory cleanup between CLIP and UNet loading, plus a scope guard to sync and release cached Burn/WGPU allocations on successful and early-error render exits.
- Implemented SD 1.5 CLIP Skip with the checkpoint's final LayerNorm applied to the selected hidden state. The helper reopens a SafeTensors view through Hearth's public `SafeTensorsFile` API to avoid adding an undeclared transitive crate dependency.
- Corrected the result-process-page text in English and Chinese: Hearth updates step progress during generation, but its current preview callback only produces an image after decode finishes.
- The image API now rejects `n` outside 1–4 and rejects `size` dimensions that overflow integer parsing instead of silently falling back to app defaults.
- Re-fetched touched source files after remote writes and checked patch anchors. No local compile/test or device test could be run because this environment cannot resolve `github.com`; the latest commit had no status checks returned by the available GitHub status query. No successful build/test is claimed.

### Remaining validation / limitations

1. Run `cargo fmt --manifest-path rust/ponko-native/Cargo.toml --all -- --check`.
2. Run `cargo test --manifest-path rust/ponko-native/Cargo.toml`.
3. With recursive Hearth/Burn/CubeCL/Cubek submodules initialized, run `./gradlew --no-daemon :app:assembleDebug`.
4. On Exynos 2400e / Xclipse 940, test SD 1.5 generation at CLIP Skip 1 and 2, a mismatched or invalid checkpoint, cancellation immediately after tapping stop, overlapping generation rejection, repeated renders for GPU-memory growth, and SDXL VAE decoding.
5. The current Hearth adapter still does not support img2img, HiRes Fix, ADetailer, UNet quantization, ResAdapter conversion, LoRA fusion, or intermediate image previews. These should remain unadvertised as active features until real implementations exist.
