// Ponko · (c) 2026 visitor257 · MIT License · github.com/visitor257/Ponko
plugins {
    id("com.android.application")
}

// Hearth is the only diffusion engine; Kotlin calls it through Rust/UniFFI.
val ponkoRustFacadeEnabled = true
val ponkoRustDir = rootProject.file("rust/ponko-native")
val ponkoRustJniLibsDir = layout.buildDirectory.dir("generated/ponkoRustJniLibs")
val ponkoUniffiKotlinDir = layout.buildDirectory.dir("generated/source/uniffi/main/java")
val ponkoCargoTargetDir = layout.buildDirectory.dir("ponko-cargo-target")

android {
    namespace = "com.litertchat.app"
    compileSdk = 36
    // Android NDK for ARM64 native libraries; diffusion inference itself is Rust.
    ndkVersion = "27.3.13750724"

    defaultConfig {
        applicationId = "com.litertchat.app"
        minSdk = 28
        targetSdk = 36
        versionCode = 14
        versionName = "1.7.1"
        ndk {
            // LiteRT-LM 与 llama.cpp 的 native 都只跑 arm64 真机，去掉 x86_64 可减小 APK
            abiFilters += "arm64-v8a"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
            // Chat runtimes may bundle libc++_shared.so.
            pickFirsts += "**/libc++_shared.so"
        }
    }

    if (ponkoRustFacadeEnabled) {
        sourceSets.getByName("main").apply {
            jniLibs.directories.add(ponkoRustJniLibsDir.get().asFile.absolutePath)
            kotlin.directories.add(ponkoUniffiKotlinDir.get().asFile.absolutePath)
            kotlin.directories.add(file("src/rust/java").absolutePath)
        }
    }
}

dependencies {
    // Google LiteRT-LM Android API (loads & runs .litertlm models)
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.17.0")
    // Coroutines (the streaming sendMessageAsync returns a Flow)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0")
    // GGUF (llama.cpp) local model runtime — full llama.cpp API incl. KV-cache reuse
    // (net.ladenthin:llama-android AAR requires minSdk 28, native = arm64-v8a)
    implementation("net.ladenthin:llama-android:5.1.0")
    // Markdown rendering for AI replies (bold/lists/headings/code/tables/links)
    implementation("io.noties.markwon:core:4.6.2")
    implementation("io.noties.markwon:ext-tables:4.6.2")
    implementation("io.noties.markwon:ext-strikethrough:4.6.2")
    implementation("io.noties.markwon:linkify:4.6.2")
    implementation("androidx.viewpager:viewpager:1.0.0")
    // 打标（Tagger）：ONNX Runtime，运行用户导入的 WD14 类 .onnx 模型（arm64-v8a）
    // PDF 文本抽取（对话发送 PDF）：Tom Roush 的 PDFBox Android 移植
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")
    // Diffusion inference is implemented by the Rust Hearth crate and Burn/WGPU Vulkan.
    if (ponkoRustFacadeEnabled) {
        implementation("net.java.dev.jna:jna:5.17.0@aar")
    }
}


// Generate UniFFI Kotlin bindings and build Rust for Android arm64-v8a as part of the app build.
// Generated artifacts stay under app/build so no machine-specific .so is committed.
if (ponkoRustFacadeEnabled) {
    val rustNativeLibrary = ponkoRustJniLibsDir.map { it.file("arm64-v8a/libponko_native.so") }
    val rustManifest = ponkoRustDir.resolve("Cargo.toml").absolutePath

    // Hearth's pinned CubeCL has a desktop-only SPIR-V disk-cache config. Apply the Android
    // compatibility patch before Cargo compiles cubecl-wgpu for aarch64-linux-android.
    val patchHearthCubeclAndroid = tasks.register<Exec>("patchHearthCubeclAndroid") {
        group = "build"
        workingDir(rootProject.projectDir)
        description = "Disable desktop-only CubeCL SPIR-V disk cache on Android."
        commandLine("bash", rootProject.file("scripts/patch-hearth-android.sh").absolutePath)
    }

    val buildPonkoRustAndroid = tasks.register<Exec>("buildPonkoRustAndroid") {
        group = "build"
        workingDir(ponkoRustDir)
        description = "Build the Ponko Rust facade for Android arm64-v8a."
        inputs.files(fileTree(ponkoRustDir) {
            include("Cargo.toml", "build.rs", "uniffi.toml", "src/**")
        })
        inputs.dir(rootProject.file("rust/hearth"))
        outputs.file(rustNativeLibrary)
        environment("CARGO_TARGET_DIR", ponkoCargoTargetDir.get().asFile.absolutePath)
        doFirst {
            ponkoRustJniLibsDir.get().asFile.mkdirs()
            check(ponkoRustDir.resolve("Cargo.toml").isFile) {
                "Missing Rust facade manifest: $rustManifest"
            }
        }
        commandLine(
            "cargo", "ndk", "-t", "arm64-v8a",
            "-o", ponkoRustJniLibsDir.get().asFile.absolutePath,
            "build", "--release", "--manifest-path", rustManifest
        )
    }

    buildPonkoRustAndroid.configure {
        dependsOn(patchHearthCubeclAndroid)
    }

    val generatePonkoUniffiKotlin = tasks.register<Exec>("generatePonkoUniffiKotlin") {
        group = "build"
        workingDir(ponkoRustDir)
        description = "Generate Kotlin bindings from the compiled Ponko UniFFI library."
        dependsOn(buildPonkoRustAndroid)
        inputs.file(rustNativeLibrary)
        inputs.file(ponkoRustDir.resolve("uniffi.toml"))
        outputs.dir(ponkoUniffiKotlinDir)
        environment("CARGO_TARGET_DIR", ponkoCargoTargetDir.get().asFile.absolutePath)
        doFirst {
            val library = rustNativeLibrary.get().asFile
            check(library.isFile) { "Rust library was not produced: $library" }
            ponkoUniffiKotlinDir.get().asFile.mkdirs()
        }
        commandLine(
            "cargo", "run", "--manifest-path", rustManifest,
            "--features", "bindgen", "--bin", "uniffi-bindgen", "--",
            "generate", "--library", rustNativeLibrary.get().asFile.absolutePath,
            "--language", "kotlin", "--out-dir", ponkoUniffiKotlinDir.get().asFile.absolutePath
        )
    }

    // The generated jniLibs directory is an output of buildPonkoRustAndroid.
    // AGP 9 validates mergeDebugJniLibFolders before mergeDebugNativeLibs, so both
    // folder merging and final native-library merging must depend on the Rust build.
    tasks.matching {
        it.name == "mergeDebugJniLibFolders" ||
        it.name == "mergeReleaseJniLibFolders" ||
        it.name == "mergeDebugNativeLibs" ||
        it.name == "mergeReleaseNativeLibs"
    }.configureEach {
        dependsOn(buildPonkoRustAndroid)
    }

    tasks.matching {
        it.name == "compileDebugKotlin" || it.name == "compileReleaseKotlin"
    }.configureEach {
        dependsOn(generatePonkoUniffiKotlin)
    }
}
