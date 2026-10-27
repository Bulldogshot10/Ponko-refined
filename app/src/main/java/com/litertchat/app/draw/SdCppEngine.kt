// Ponko · (c) 2026 visitor257 · MIT License · github.com/visitor257/Ponko
package com.litertchat.app.draw

import android.graphics.Bitmap

/**
 * Ponko 自编 stable-diffusion.cpp（GGUF）引擎。
 *
 * 相比 llmedge，这里能直接控制：线程数、采样器、调度器、LoRA、量化类型、取消。
 */
object SdCppEngine {

    @Volatile
    private var lastCreateErrorValue: String? = null

    fun lastCreateError(): String? = lastCreateErrorValue

    init {
        // Rust/UniFFI owns the Kotlin-facing API; ponko_sd.so exports a plain C ABI.
        // Load the C++ bridge first so Rust can resolve it through libloading.
        System.loadLibrary("c++_shared")
        System.loadLibrary("ponko_sd")
    }

    /** 采样器（对应 sd.cpp enum sample_method_t） */
    enum class Sampler(val code: Int, val label: String) {
        EULER(0, "Euler"),
        EULER_A(1, "Euler a"),
        HEUN(2, "Heun"),
        DPM2(3, "DPM2"),
        DPMPP2S_A(4, "DPM++ 2S a"),
        DPMPP2M(5, "DPM++ 2M"),
        DPMPP2M_V2(6, "DPM++ 2M v2"),
        IPNDM(7, "IPNDM"),
        IPNDM_V(8, "IPNDM V"),
        LCM(9, "LCM"),
        DDIM_TRAILING(10, "DDIM"),
        TCD(11, "TCD"),
        EULER_CFG_PP(15, "Euler CFG++"),
        EULER_A_CFG_PP(16, "Euler a CFG++"),

        // 以下 7 项 sd.cpp 早已实现（见 runtime/denoiser.hpp 的分发），此前只是 UI 没暴露。
        // 一律追加在末尾：下拉框按声明顺序生成，旧用户偏好里存的是索引，插在中间会错位。
        RES_MULTISTEP(12, "Res Multistep"),
        RES_2S(13, "Res 2S"),
        ER_SDE(14, "ER SDE"),
        EULER_GE(17, "Euler GE"),
        DPMPP2M_SDE(18, "DPM++ 2M SDE"),
        DPMPP2M_SDE_BT(19, "DPM++ 2M SDE B&T"),
        LMS(20, "LMS"),

        // Appended to preserve existing persisted spinner indices. Both use
        // the native DPM++ SDE implementation; the second is a Karras preset.
        DPMPP_SDE(21, "DPM++ SDE"),
        DPMPP_SDE_KARRAS(21, "DPM++ SDE Karras"),
    }

    /** 调度器（对应 sd.cpp enum scheduler_t） */
    enum class Scheduler(val code: Int, val label: String) {
        DISCRETE(0, "Discrete"),
        KARRAS(1, "Karras"),
        EXPONENTIAL(2, "Exponential"),
        AYS(3, "AYS"),
        SGM_UNIFORM(5, "SGM Uniform"),
        SIMPLE(6, "Simple"),
        SMOOTHSTEP(7, "Smoothstep"),
        KL_OPTIMAL(8, "KL Optimal"),
        LCM(9, "LCM"),
        BETA(15, "Beta"),
        TCD(16, "TCD"),
    }

    enum class HiresUpscaler(val code: Int, val label: String) {
        NONE(0, "None"), LATENT(1, "Latent"), LATENT_NEAREST(2, "Latent (nearest)"),
        LATENT_NEAREST_EXACT(3, "Latent (nearest-exact)"), LATENT_ANTIALIASED(4, "Latent (antialiased)"),
        LATENT_BICUBIC(5, "Latent (bicubic)"), LATENT_BICUBIC_ANTIALIASED(6, "Latent (bicubic antialiased)"),
        LANCZOS(7, "Lanczos"), NEAREST(8, "Nearest"), MODEL(9, "Model"),
    }

    data class HiResSettings(
        val enabled: Boolean = false,
        val upscaler: HiresUpscaler = HiresUpscaler.LATENT,
        val modelPath: String? = null,
        val scale: Float = 2.0f,
        val targetWidth: Int = 0,
        val targetHeight: Int = 0,
        val steps: Int = 0,
        val denoisingStrength: Float = 0.7f,
        val tileSize: Int = 128,
    )

    /** 权重类型（对应 sd.cpp enum sd_type_t）；KEEP = 保持模型原样 */
    const val WTYPE_KEEP = -1
    const val WTYPE_F32 = 0
    const val WTYPE_F16 = 1
    const val WTYPE_Q4_0 = 2
    const val WTYPE_Q4_1 = 3
    const val WTYPE_Q5_0 = 6
    const val WTYPE_Q5_1 = 7
    const val WTYPE_Q8_0 = 8

    fun interface StepCallback {
        fun onStep(step: Int, steps: Int)
    }

    /** 生成过程预览：每一步回调一张低分辨率小图（RGB888，行优先，左上原点） */
    fun interface PreviewCallback {
        fun onPreview(step: Int, width: Int, height: Int, rgb: ByteArray)
    }

    private val nativeEngine: com.litertchat.app.ponko.NativeEngine by lazy {
        // JNA loads libponko_native.so from UniFFI. Load the C ABI bridge first so
        // Rust can resolve its stable-diffusion.cpp symbols through libloading.
        com.litertchat.app.ponko.NativeEngine()
    }

    fun nativeInfo(): String = runCatching { nativeEngine.info() }.getOrDefault("(n/a)")

    /** Runtime Vulkan capabilities detected by ggml-vulkan for the active GPU. */
    fun nativeVulkanDiagnostics(): String =
        runCatching { nativeEngine.vulkanDiagnostics() }.getOrDefault("")

    /** Quantize only model.diffusion_model.* tensors; retain source validation and atomic output. */
    fun nativeQuantizeUnet(
        inputPath: String,
        outputPath: String,
        quantType: String,
        diffusionSlot: Boolean,
        nThreads: Int,
    ): Boolean = runCatching {
        nativeEngine.quantizeUnet(
            com.litertchat.app.ponko.QuantizeUnetRequest(
                inputPath, outputPath, quantType, diffusionSlot, nThreads
            )
        )
        true
    }.getOrDefault(false)

    /** Fuse one or more LoRA adapters into a new standalone GGUF checkpoint. */
    fun nativeFuseLoras(
        inputPath: String,
        loraPaths: List<String>,
        loraScales: List<Float>,
        outputPath: String,
        quantType: String = "f16",
        nThreads: Int,
    ): Boolean = runCatching {
        nativeEngine.fuseLoras(
            com.litertchat.app.ponko.LoraFusionRequest(
                inputPath, loraPaths, loraScales, outputPath, quantType, nThreads
            )
        )
        true
    }.getOrDefault(false)

    /** Merge ResAdapter normalization tensors and quantize the UNet into a new GGUF. */
    fun nativeApplyResAdapter(
        inputModelPath: String,
        vaePath: String?,
        normalizationPath: String,
        outputPath: String,
        quantType: String,
        nThreads: Int,
    ): Boolean = runCatching {
        nativeEngine.applyResadapter(
            com.litertchat.app.ponko.ResAdapterRequest(
                inputModelPath, vaePath, normalizationPath, outputPath, quantType, nThreads
            )
        )
        true
    }.getOrDefault(false)

    /** Convert a supported PyTorch .bin state_dict to safetensors, surfacing native errors. */
    fun nativeConvertTorchBinToSafetensors(inputPath: String, outputPath: String): String? = try {
        nativeEngine.convertTorchBinToSafetensors(inputPath, outputPath)
        null
    } catch (error: Throwable) {
        error.message ?: error.javaClass.simpleName
    }

    fun vulkanDiagnostics(): String =
        runCatching { nativeVulkanDiagnostics() }.getOrDefault("")

    /** Last new_sd_ctx parameter dump, useful when diagnosing model-load failures. */
    fun nativeLastParams(): String = runCatching { nativeEngine.lastParams() }.getOrDefault("")

    private fun modelSlots(slots: Array<String?>?): com.litertchat.app.ponko.ModelSlots? {
        if (slots == null) return null
        fun slot(index: Int): String? = slots.getOrNull(index)?.takeIf { it.isNotBlank() }
        return com.litertchat.app.ponko.ModelSlots(
            slot(0), slot(1), slot(2), slot(3), slot(4), slot(5), slot(6),
            slot(7), slot(8), slot(9), slot(10), slot(11), slot(12),
            slot(13), slot(14), slot(15)
        )
    }

    /** Model loading is exposed through a typed UniFFI request, not a positional JNI signature. */
    fun nativeCreate(
        modelPath: String,
        vaePath: String?,
        nThreads: Int,
        wtype: Int,
        flashAttn: Boolean,
        vaeConvDirect: Boolean,
        backend: String?,
        slots: Array<String?>?,
    ): Long {
        lastCreateErrorValue = null
        return try {
            nativeEngine.createContext(
                com.litertchat.app.ponko.ModelLoadRequest(
                    modelPath, vaePath, nThreads, wtype, flashAttn, vaeConvDirect, backend, modelSlots(slots)
                )
            ).also { handle ->
                if (handle == 0L) lastCreateErrorValue = "Native model context returned an empty handle"
            }
        } catch (error: Throwable) {
            lastCreateErrorValue = error.message ?: error.javaClass.simpleName
            0L
        }
    }

    data class Lora(val path: String, val scale: Float)

    data class ADetailerSettings(
        val enabled: Boolean = false,
        val detectorPath: String? = null,
        val prompt: String = "",
        val negativePrompt: String = "",
        val confidence: Float = 0.3f,
        val inpaintSize: Int = 512,
        val padding: Int = 32,
        val maskBlur: Int = 4,
        val denoisingStrength: Float = 0.4f,
        /** 0 inherits the main generation step count. */
        val steps: Int = 0,
    ) {
        val extraArgs: String
            get() = listOf(
                "input_size=640",
                "confidence=$confidence",
                "inpaint_padding=$padding",
                "inpaint_width=$inpaintSize",
                "inpaint_height=$inpaintSize",
                "mask_blur=$maskBlur",
                "denoising_strength=$denoisingStrength",
                "steps=$steps",
            ).joinToString(",")
    }

    /**
     * Kotlin keeps the existing API shape while UniFFI carries a typed request through Rust.
     * RGB888 is converted back to the legacy width/height-prefixed ARGB array so DrawPage and
     * the API server continue to use the same Bitmap conversion path.
     */
    fun nativeGenerate(
        handle: Long,
        prompt: String,
        negative: String,
        loraPaths: Array<String>,
        loraScales: FloatArray,
        width: Int,
        height: Int,
        steps: Int,
        cfg: Float,
        seed: Long,
        sampleMethod: Int,
        scheduler: Int,
        eta: Float,
        hiresEnabled: Boolean,
        hiresUpscaler: Int,
        hiresModelPath: String,
        hiresScale: Float,
        hiresTargetWidth: Int,
        hiresTargetHeight: Int,
        hiresSteps: Int,
        hiresDenoisingStrength: Float,
        hiresTileSize: Int,
        clipSkip: Int,
        cb: StepCallback?,
        pcb: PreviewCallback?,
        initData: ByteArray?,
        initWidth: Int,
        initHeight: Int,
        strength: Float,
        adetailer: ADetailerSettings = ADetailerSettings(),
        controlImage: ByteArray? = null,
        controlImageWidth: Int = 0,
        controlImageHeight: Int = 0,
        controlStrength: Float = 0.9f,
        ipAdapterImage: ByteArray? = null,
        ipAdapterImageWidth: Int = 0,
        ipAdapterImageHeight: Int = 0,
        ipAdapterStrength: Float = 1.0f,
    ): IntArray? = runCatching {
        val loraCount = minOf(loraPaths.size, loraScales.size)
        val request = com.litertchat.app.ponko.GenerationRequest(
            prompt,
            negative,
            loraPaths.take(loraCount),
            loraScales.take(loraCount).toList(),
            width,
            height,
            steps,
            cfg,
            seed,
            sampleMethod,
            scheduler,
            eta,
            hiresEnabled,
            hiresUpscaler,
            hiresModelPath.takeIf { it.isNotEmpty() },
            hiresScale,
            hiresTargetWidth,
            hiresTargetHeight,
            hiresSteps,
            hiresDenoisingStrength,
            hiresTileSize,
            clipSkip,
            initData,
            initWidth,
            initHeight,
            strength,
            adetailer.enabled,
            adetailer.detectorPath,
            adetailer.prompt,
            adetailer.negativePrompt,
            adetailer.extraArgs,
            controlImage,
            controlImageWidth,
            controlImageHeight,
            controlStrength,
            ipAdapterImage,
            ipAdapterImageWidth,
            ipAdapterImageHeight,
            ipAdapterStrength,
        )
        val stepProxy = cb?.let { callback ->
            object : com.litertchat.app.ponko.StepCallback {
                override fun onStep(step: Int, steps: Int) = callback.onStep(step, steps)
            }
        }
        val previewProxy = pcb?.let { callback ->
            object : com.litertchat.app.ponko.PreviewCallback {
                override fun onPreview(step: Int, width: Int, height: Int, rgb888: ByteArray) {
                    callback.onPreview(step, width, height, rgb888)
                }
            }
        }
        val image = com.litertchat.app.ponko.generateImage(nativeEngine, handle, request, stepProxy, previewProxy)
        val pixelCount = image.width.toLong() * image.height.toLong()
        require(image.width > 0 && image.height > 0 && pixelCount <= Int.MAX_VALUE - 2)
        require(image.rgb888.size.toLong() == pixelCount * 3L)
        val pixels = IntArray(pixelCount.toInt() + 2)
        pixels[0] = image.width
        pixels[1] = image.height
        for (i in 0 until pixelCount.toInt()) {
            val j = i * 3
            val r = image.rgb888[j].toInt() and 0xFF
            val g = image.rgb888[j + 1].toInt() and 0xFF
            val b = image.rgb888[j + 2].toInt() and 0xFF
            pixels[i + 2] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        pixels
    }.getOrNull()

    fun nativeCancel(handle: Long) {
        runCatching { nativeEngine.cancel(handle) }
    }

    /** Backend devices in "name<TAB>description" format, preserving the old UI contract. */
    fun nativeListDevices(): String = runCatching { nativeEngine.listDevices() }.getOrDefault("")

    fun nativeFree(handle: Long) {
        if (handle != 0L) runCatching { nativeEngine.freeContext(handle) }
    }

    fun info(): String = runCatching { nativeInfo() }.getOrDefault("(n/a)")

    /** 最近一次 new_sd_ctx 用的参数（多行文本） */
    fun lastParams(): String = runCatching { nativeLastParams() }.getOrDefault("")

    /** 后端设备清单（名字 -> 描述）；取不到时返回空表。 */
    fun listDevices(): List<Pair<String, String>> =
        runCatching { nativeListDevices() }.getOrDefault("")
            .lineSequence()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .map { line ->
                val i = line.indexOf('\t')
                if (i < 0) line to "" else line.substring(0, i) to line.substring(i + 1)
            }
            .toList()

    /** 是否有可用的 Vulkan(GPU) 设备；有则返回设备名（如 "Vulkan0"）。 */
    fun vulkanDevice(): String? =
        listDevices().firstOrNull { it.first.contains("vulkan", true) }?.first

    /** 加载模型并返回 handle；0 表示失败。
     *
     *  flashAttn：CLIP 与 UNet 的 FlashAttention。实测不开会让每步慢约 28%，默认开。
     *  backend：ggml 后端/设备名（"CPU" / "Vulkan0"）；null = 交给 sd.cpp 自动挑。
     */
    fun create(
        modelPath: String,
        vaePath: String?,
        nThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 4),
        wtype: Int = WTYPE_KEEP,
        flashAttn: Boolean = true,
        vaeConvDirect: Boolean = true,
        backend: String? = null,
        slots: Array<String?>? = null,
    ): Long = nativeCreate(modelPath, vaePath, nThreads, wtype, flashAttn, vaeConvDirect, backend, slots)

    /**
     * 槽位顺序（与 JNI 里的 SlotIndex 一一对应；sd.cpp 的 sd_ctx_params_t 路径字段）。
     * 单文件模型只用 "model"；多文件模型按家族填 diffusion / clip_l / t5xxl / llm / vae 等。
     */
    val SLOT_KEYS = arrayOf(
        "model", "diffusion", "high_noise", "uncond",
        "clip_l", "clip_g", "clip_vision", "t5xxl",
        "llm", "llm_vision", "vae", "audio_vae", "taesd",
        "control_net", "ip_adapter", "embedding",
    )

    /** 把槽位表（key -> 文件绝对路径）转成 JNI 要的定长数组 */
    fun buildSlots(map: Map<String, String>): Array<String?> {
        val out = arrayOfNulls<String>(SLOT_KEYS.size)
        for ((k, v) in map) {
            val i = SLOT_KEYS.indexOf(k)
            if (i >= 0) out[i] = v
        }
        return out
    }

    /** 生成一张图；失败返回 null。
     *
     *  initImage 非空则会走图生图（img2img）：它会被缩放到 width×height 后作为初始 latent，
     *  strength(<1) 决定从第几步开始去噪（越小越接近原图）。
     */
    fun render(
        handle: Long,
        prompt: String,
        negative: String = "",
        loras: List<Lora> = emptyList(),
        width: Int = 512,
        height: Int = 512,
        steps: Int = 20,
        cfg: Float = 7.0f,
        clipSkip: Int = 1,
        seed: Long = -1L,
        sampler: Sampler = Sampler.EULER_A,
        scheduler: Scheduler = Scheduler.DISCRETE,
        eta: Float = -1.0f,
        hires: HiResSettings = HiResSettings(),
        cb: StepCallback? = null,
        onPreview: PreviewCallback? = null,
        initImage: ByteArray? = null,
        initWidth: Int = 0,
        initHeight: Int = 0,
        strength: Float = 1.0f,
        adetailer: ADetailerSettings = ADetailerSettings(),
        controlImage: ByteArray? = null,
        controlImageWidth: Int = 0,
        controlImageHeight: Int = 0,
        controlStrength: Float = 0.9f,
        ipAdapterImage: ByteArray? = null,
        ipAdapterImageWidth: Int = 0,
        ipAdapterImageHeight: Int = 0,
        ipAdapterStrength: Float = 1.0f,
    ): Bitmap? {
        val effectiveSeed = if (seed < 0) (System.currentTimeMillis() % 1_000_000_000L) else seed
        val useInit = initImage != null && initImage.isNotEmpty() && initWidth > 0 && initHeight > 0
        val loraPaths = loras.map { it.path }.toTypedArray()
        val loraScales = loras.map { it.scale }.toFloatArray()
        val px = nativeGenerate(
            handle, prompt, negative, loraPaths, loraScales,
            width, height, steps, cfg, effectiveSeed,
            sampler.code, scheduler.code, eta,
            hires.enabled, hires.upscaler.code, hires.modelPath.orEmpty(),
            hires.scale, hires.targetWidth, hires.targetHeight, hires.steps,
            hires.denoisingStrength, hires.tileSize,
            clipSkip.coerceIn(1, 12), cb, onPreview,
            if (useInit) initImage else null,
            if (useInit) initWidth else 0,
            if (useInit) initHeight else 0,
            if (useInit) strength else 1.0f,
            adetailer,
            controlImage,
            controlImageWidth,
            controlImageHeight,
            controlStrength,
            ipAdapterImage,
            ipAdapterImageWidth,
            ipAdapterImageHeight,
            ipAdapterStrength,
        ) ?: return null
        if (px.size < 2) return null
        val outWidth = px[0]
        val outHeight = px[1]
        if (outWidth <= 0 || outHeight <= 0 || px.size - 2 != outWidth * outHeight) return null
        return Bitmap.createBitmap(px.copyOfRange(2, px.size), outWidth, outHeight, Bitmap.Config.ARGB_8888)
    }

        /** 预览小图（RGB888）→ Bitmap（ARGB_8888）；非法输入返回 null */
    fun rgbToBitmap(width: Int, height: Int, rgb: ByteArray): Bitmap? {
        if (width <= 0 || height <= 0 || rgb.size < width * height * 3) return null
        val px = IntArray(width * height)
        var j = 0
        for (i in px.indices) {
            px[i] = (0xFF shl 24) or ((rgb[j].toInt() and 0xFF) shl 16) or
                    ((rgb[j + 1].toInt() and 0xFF) shl 8) or (rgb[j + 2].toInt() and 0xFF)
            j += 3
        }
        return Bitmap.createBitmap(px, width, height, Bitmap.Config.ARGB_8888)
    }

/**
     * Bitmap → RGB888 交错字节（每像素 3 字节，行优先，左上原点）。
     * sd.cpp 的 init_image 就吃这个布局（见 sd_image_get_f32）。
     */
    fun bitmapToRgb888(bitmap: Bitmap): ByteArray {
        val w = bitmap.width
        val h = bitmap.height
        val px = IntArray(w * h)
        bitmap.getPixels(px, 0, w, 0, 0, w, h)
        val out = ByteArray(w * h * 3)
        var j = 0
        for (i in px.indices) {
            val p = px[i]
            out[j++] = ((p shr 16) and 0xFF).toByte()
            out[j++] = ((p shr 8) and 0xFF).toByte()
            out[j++] = (p and 0xFF).toByte()
        }
        return out
    }
}
