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
