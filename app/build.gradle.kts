// Ponko · (c) 2026 visitor257 · MIT License · github.com/visitor257/Ponko
plugins {
    id("com.android.application")
}

// Set -PponkoEnableVulkan=OFF for a CPU-only variant; default stays ON.
// Keep the requested Exynos ISA profile enabled unless explicitly overridden.
val ponkoEnableVulkan = providers.gradleProperty("ponkoEnableVulkan").orElse("ON").get()
val ponkoExynosCpuProfile = providers.gradleProperty("ponkoExynosCpuProfile").orElse("ON").get()
val ponkoCpuArchFlags = if (ponkoExynosCpuProfile.equals("ON", ignoreCase = true)) {
    "-march=armv8.6-a+dotprod+i8mm+fp16 -mtune=cortex-x4"
} else {
    "-march=armv8.2-a+dotprod+fp16"
}
val ponkoEnableCompilerCache = providers.gradleProperty("ponkoEnableCompilerCache").orElse("ON").get()
// Rust/UniFFI is the required Kotlin-facing native API.
val ponkoRustFacadeEnabled = true
val ponkoRustDir = rootProject.file("rust/ponko-native")
val ponkoRustJniLibsDir = layout.buildDirectory.dir("generated/ponkoRustJniLibs")
val ponkoUniffiKotlinDir = layout.buildDirectory.dir("generated/source/uniffi/main/java")
val ponkoCargoTargetDir = layout.buildDirectory.dir("ponko-cargo-target")

android {
    namespace = "com.litertchat.app"
    compileSdk = 36
    // 自编 stable-diffusion.cpp 所用的 NDK（须与 jniLibs 里的 libomp.so 同源）
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
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DPONKO_ENABLE_VULKAN=${ponkoEnableVulkan}",
                    "-DPONKO_EXYNOS_2400E_CPU_PROFILE=${ponkoExynosCpuProfile}",
                    "-DPONKO_ENABLE_COMPILER_CACHE=${ponkoEnableCompilerCache}",
                )
            }
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

    // 自编 stable-diffusion.cpp：源码在 src/main/cpp（含内联 ggml），
    // 每次 assembleRelease 由 NDK 现场编译，产物直接进 APK，不再手工往 jniLibs 拷贝。
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
            // 自编的 libstable-diffusion.so（+ libponko_sd.so / libomp.so）依赖它
            pickFirsts += "**/libc++_shared.so"
        }
    }

    if (ponkoRustFacadeEnabled) {
        sourceSets.getByName("main").apply {
            jniLibs.directories.add(ponkoRustJniLibsDir.get().asFile.absolutePath)
            java.directories.add(ponkoUniffiKotlinDir.get().asFile.absolutePath)
            java.directories.add(file("src/rust/java").absolutePath)
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
    // 绘图（文生图）：自编的 stable-diffusion.cpp（arm64-v8a），源码在 src/main/cpp，
    // 由 NDK/CMake 现场编译出 libstable-diffusion.so + libponko_sd.so（C ABI bridge，由 Rust 调用）。
    // 相比现成 AAR，这里能直接控制线程数 / 采样器 / 调度器 / LoRA / 量化类型 / 取消。
    if (ponkoRustFacadeEnabled) {
        implementation("net.java.dev.jna:jna:5.17.0@aar")
    }
}


// Generate UniFFI Kotlin bindings and build Rust for Android arm64-v8a as part of the app build.
// Generated artifacts stay under app/build so no machine-specific .so is committed.
if (ponkoRustFacadeEnabled) {
    val rustNativeLibrary = ponkoRustJniLibsDir.map { it.file("arm64-v8a/libponko_native.so") }
    val rustManifest = ponkoRustDir.resolve("Cargo.toml").absolutePath

    val buildPonkoRustAndroid = tasks.register<Exec>("buildPonkoRustAndroid") {
        group = "build"
        workingDir(ponkoRustDir)
        description = "Build the Ponko Rust facade for Android arm64-v8a."
        inputs.files(fileTree(ponkoRustDir) {
            include("Cargo.toml", "build.rs", "uniffi.toml", "src/**")
        })
        outputs.file(rustNativeLibrary)
        environment("CARGO_TARGET_DIR", ponkoCargoTargetDir.get().asFile.absolutePath)
        environment("PONKO_CPU_ARCH_FLAGS", ponkoCpuArchFlags)
        environment("PONKO_VULKAN_ENABLED", ponkoEnableVulkan)
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

    val generatePonkoUniffiKotlin = tasks.register<Exec>("generatePonkoUniffiKotlin") {
        group = "build"
        workingDir(ponkoRustDir)
        description = "Generate Kotlin bindings from the compiled Ponko UniFFI library."
        dependsOn(buildPonkoRustAndroid)
        inputs.file(rustNativeLibrary)
        inputs.file(ponkoRustDir.resolve("uniffi.toml"))
        outputs.dir(ponkoUniffiKotlinDir)
        environment("CARGO_TARGET_DIR", ponkoCargoTargetDir.get().asFile.absolutePath)
        environment("PONKO_CPU_ARCH_FLAGS", ponkoCpuArchFlags)
        environment("PONKO_VULKAN_ENABLED", ponkoEnableVulkan)
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

    tasks.matching {
        it.name == "mergeDebugNativeLibs" || it.name == "mergeReleaseNativeLibs"
    }.configureEach {
        dependsOn(buildPonkoRustAndroid)
    }

    tasks.matching {
        it.name == "compileDebugKotlin" || it.name == "compileReleaseKotlin"
    }.configureEach {
        dependsOn(generatePonkoUniffiKotlin)
    }
}
