# Ponko Hearth native engine

This branch fully replaces the drawing inference path with Hearth. There is no stable-diffusion.cpp engine, GGML drawing backend, C ABI C++ bridge, or alternate diffusion backend in the Android build.

## Current inference scope

- SD 1.5 and SDXL checkpoints in SafeTensors format.
- Bundled OpenAI CLIP ViT-L/14 tokenizer assets (`vocab.json` and `merges.txt`); model sets may optionally override either file.
- SD 1.5 VAE embedded in the checkpoint or imported separately.
- SDXL with a separate FP16-safe VAE.
- LoRA application to the supported CLIP and UNet layers, with per-adapter strength.
- Euler, Euler a, DPM++ 2M and DPM++ SDE samplers; Normal and Karras schedules.
- Progress callbacks and cancellation between denoising steps.
- Burn/WGPU Vulkan as the only drawing execution backend.

Chat GGUF remains on the independent llama.cpp chat runtime. That is not a second drawing backend.

## Not implemented in this Hearth adapter yet

Drawing GGUF/quantization, img2img, HiRes Fix, ADetailer, UNet quantization, ResAdapter conversion, and LoRA fusion. These are deliberately not exposed by the drawing UI until their actual Hearth implementations are available.

## Build requirements

- Rust stable with the 2024 edition.
- `aarch64-linux-android` Rust target and `cargo-ndk`.
- Android NDK 27.3.13750724, Android SDK 36, JDK 21.
- Recursive submodule checkout: `rust/hearth`, plus Hearth's Burn, CubeCL and Cubek source submodules.

## Build

```bash
cargo fmt --manifest-path rust/ponko-native/Cargo.toml --all
cargo test --manifest-path rust/ponko-native/Cargo.toml
./gradlew :app:assembleDebug
```

Gradle compiles the Hearth adapter into `libponko_native.so` and generates UniFFI Kotlin bindings. Drawing inference does not use CMake and does not emit or package `libponko_sd.so` or `libstable-diffusion.so`.
