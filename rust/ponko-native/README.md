# Ponko Rust / UniFFI facade

This is the first, compatibility-preserving migration slice for the Android native boundary.

## Current status

- Defines typed UniFFI records matching the current model-slot and image-generation fields.
- Exposes API-version/build-profile introspection and request preflight validation.
- Keeps CFG = 0 valid for TCD-style workflows and does not introduce a LoRA count limit.
- Validates the three quantization types currently accepted by the C++ converter: `q4_0`, `q5_0`, and `q8_0`.
- Does **not** replace or remove the existing JNI engine yet. Generation, previews, cancellation, quantization, ResAdapter, model loading, Vulkan diagnostics, device listing, and native-handle ownership remain on the legacy path until parity tests cover them.

Keeping that path intact is deliberate: this branch must not lose any existing Ponko feature while the Rust boundary is validated.

## Tooling

Install Rust stable and cargo-ndk:

```sh
rustup update stable
rustup target add aarch64-linux-android
cargo install cargo-ndk
```

The Android SDK/NDK must be installed and discoverable from the environment. The app's current build uses NDK `27.3.13750724`.

## Android build with the opt-in facade

From the repository root:

```sh
./gradlew :app:assembleDebug -PponkoRustFacade=true
```

On Windows, use `gradlew.bat` or the existing PowerShell build workflow. This opt-in builds `libponko_native.so`, generates Kotlin bindings with UniFFI, packages the native library, and adds JNA for the stable Kotlin binding backend. The default build is unchanged until parity testing is complete.

## Unit tests

```sh
cargo test --manifest-path rust/ponko-native/Cargo.toml
```

## Cutover rule

Do not delete `sdcpp_jni.cpp` or alter `SdCppEngine`'s existing native calls until each current API has a Rust counterpart and parity coverage: create/free, text-to-image, img2img, upscaling, sampler/scheduler/ETA, multiple LoRAs, progress/preview callbacks, cancellation, model slots, quantization validation/conversion, ResAdapter, diagnostics, and device enumeration.
