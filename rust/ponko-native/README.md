# Ponko Rust / UniFFI native facade

Kotlin now calls the drawing engine through generated UniFFI bindings into Rust. Rust owns the app-facing API, validation, callback routing, model-handle registry, and lifecycle synchronization. The inference implementation remains stable-diffusion.cpp + GGML, exposed by the plain C ABI in `app/src/main/cpp/ponko_sd_c_api.h` and `ponko_sd_c_api.cpp`.

## Native API coverage

The facade covers model context creation and release, text-to-image and img2img, all existing sampler/scheduler/ETA controls, caller-provided LoRA sets, HiRes settings, CLIP skip, optional ControlNet and IP-Adapter image conditioning, one optional Textual Inversion embedding for CLIP-based SD models, progress/preview callbacks, cancellation, Vulkan diagnostics, last-parameter logging, backend device enumeration, UNet quantization, and ResAdapter conversion. Existing Kotlin call sites keep their `SdCppEngine` wrapper names, but those wrappers now call the generated UniFFI API rather than hand-written JNI methods.

The native handle registry blocks context destruction while generate/cancel calls are using a context. Generation remains single-flight because stable-diffusion.cpp's progress and preview callbacks are registered globally.

See [API_PARITY.md](API_PARITY.md) for the explicit mapping between the old API and the new Rust/C ABI implementation.

## Tooling

Install Rust stable and the Android ARM64 target, plus cargo-ndk:

```sh
rustup update stable
rustup target add aarch64-linux-android
cargo install cargo-ndk --locked
```

The Android SDK/NDK must be installed and discoverable. Ponko currently uses NDK `27.3.13750724`, CMake `3.31.6`, and builds only `arm64-v8a`.

## Build and test

From the repository root:

```sh
cargo fmt --manifest-path rust/ponko-native/Cargo.toml --all -- --check
cargo test --manifest-path rust/ponko-native/Cargo.toml
./gradlew :app:assembleDebug
```

On Windows, use `gradlew.bat`. Gradle builds `libponko_native.so` for Android, generates UniFFI Kotlin bindings, and packages them with `libponko_sd.so` and `libstable-diffusion.so`. Install cargo-ndk before building.

The app-facing JNI bridge has been removed from the CMake target. `libponko_sd.so` is a C ABI adapter loaded by Rust, not a library of Java native methods.
