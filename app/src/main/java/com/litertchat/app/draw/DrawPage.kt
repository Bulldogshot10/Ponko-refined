// Ponko · (c) 2026 visitor257 · MIT License · github.com/visitor257/Ponko
package com.litertchat.app.draw

import com.litertchat.app.DrawCore
import com.litertchat.app.PonkoTheme
import com.litertchat.app.R

import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.net.Uri
import android.provider.OpenableColumns
import android.text.InputType
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.ScrollView
import android.widget.Toast
import androidx.viewpager.widget.PagerAdapter
import androidx.viewpager.widget.ViewPager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/** 用户主动中断生成时抛出（与真正的失败区分开）。消息不面向用户，用固定英文。 */
class GenerationCancelledException : RuntimeException("Generation cancelled")

/**
 * 绘图页（文生图）。
 *
 * 绘图引擎为 **Hearth + Burn/WGPU Vulkan**，读取支持的 SafeTensors checkpoint
 * （支持 SD 1.5 与 SDXL；每个模型集需要 tokenizer 的 vocab.json / merges.txt）。
 *
 * 加速有两条**互相独立**的线，可以叠加：
 *  - **量化**：由模型文件本身决定（Q4_0 每步比 Q8_0 便宜不少）。App 只读取并展示，不能凭空转换。
 *  - **LoRA**：挂 LCM-LoRA 这类「少步蒸馏补丁」，把 20 步压到 4~8 步（约 5 倍）。
 *
 * LoRA 文件放在 filesDir/draw/lora/，通过 prompt 里的 `lora:名字:权重` 语法激活。
 */
class DrawPage(
    private val act: Activity,
    private val scope: CoroutineScope,
    /**
     * 配色表（浅色 / 深色由 MainActivity 决定）。
     * 下面三个只是它的便捷取用点；切换主题时整个 Activity 重建，页面随之重画。
     */
    pal: PonkoTheme,
) {
    /** 配色表；换主题时先 setPalette 再 build() 重建本页视图（native 句柄等字段都保留） */
    var pal: PonkoTheme = pal
        private set

    private val primary: Int get() = pal.primary
    private val textColor: Int get() = pal.text
    private val subText: Int get() = pal.subText

    /** 换配色（下次 build() 生效） */
    fun setPalette(p: PonkoTheme) { pal = p }

    /** 是否正在出图（换主题要重建视图，出图途中不适合重建） */
    fun isBusy(): Boolean = generating

    /** CPU worker count used for the next native drawing-model load. */
    fun cpuThreads(): Int = nThreads

    fun setCpuThreads(value: Int) {
        val maxThreads = Runtime.getRuntime().availableProcessors().coerceIn(1, 12)
        nThreads = value.coerceIn(1, maxThreads)
        prefs().edit().putInt(KEY_CPU_THREADS, nThreads).apply()
    }

    fun taesdEnabled(): Boolean = prefs().getBoolean(KEY_TAESD_ENABLED, false)

    fun setTaesdEnabled(enabled: Boolean) {
        prefs().edit().putBoolean(KEY_TAESD_ENABLED, enabled).apply()
    }

    /** Enable GGML's direct Conv2D path for the original VAE; applies on the next model load. */
    fun vaeConvDirectEnabled(): Boolean = prefs().getBoolean(KEY_VAE_CONV_DIRECT, true)

    fun setVaeConvDirectEnabled(enabled: Boolean) {
        prefs().edit().putBoolean(KEY_VAE_CONV_DIRECT, enabled).apply()
    }

    fun taesdFileName(): String? = activeSetOrSelected()?.files?.get("taesd")?.name

    fun hasTaesdFile(): Boolean = activeSetOrSelected()?.files?.get("taesd")?.isFile == true

    companion object {
        const val DIR_NAME = "draw"
        private const val REQ_IMAGE = 0x5D02
        private const val REQ_RESADAPTER_LORA = 0x5D04
        private const val REQ_RESADAPTER_NORM = 0x5D05
        private const val REQ_HIRES_MODEL = 0x5D06
        private const val REQ_ADETAILER_MODEL = 0x5D07
        private const val REQ_LORA_FUSION_IMPORT = 0x5D08
        private const val HIRES_UPSCALER_DIR = "hires-upscalers"
        private const val ADETAILER_MODEL_DIR = "adetailer-models"

        /** LCM-LoRA：HuggingFace 官方仓库路径（文件实名为 pytorch_lora_weights.safetensors） */
        private const val LORA_HF_PATH =
            "latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors"

        /** 下载后重命名成这个，prompt 里就用 `lora:lcm-lora-sdv1-5:1` 引用 */
        const val LORA_NAME = "lcm-lora-sdv1-5"

        /** 模型集偏好键（当前选中的模型集 id） */
        private const val KEY_SEL_SET = "selSet"
        private const val KEY_CPU_THREADS = "drawCpuThreads"
        private const val KEY_TAESD_ENABLED = "drawTaesdEnabled"
        private const val KEY_VAE_CONV_DIRECT = "drawVaeConvDirect"

        /** 单文件 / 多文件 */
        const val KIND_SINGLE = "single"
        const val KIND_MULTI = "multi"

        /** 家族 id */
        const val FAM_SD15 = "sd15"
        const val FAM_SDXL = "sdxl"
        const val FAM_SD3 = "sd3"
        const val FAM_FLUX = "flux"
        const val FAM_QWEN = "qwen"
        const val FAM_HIDREAM = "hidream"
        const val FAM_VIDEO = "video"
        const val FAM_OTHER = "other"

        /** 家族显示名 */
        fun familyLabelRes(family: String): Int = when (family) {
            FAM_SD15 -> R.string.s_337
            FAM_SDXL -> R.string.s_338
            FAM_SD3 -> R.string.s_339
            FAM_FLUX -> R.string.s_340
            FAM_QWEN -> R.string.s_341
            FAM_HIDREAM -> R.string.s_342
            FAM_VIDEO -> R.string.s_343
            else -> R.string.s_344
        }

        /** 槽位显示名 */
        fun slotLabelRes(key: String): Int = when (key) {
            "model" -> R.string.s_347
            "diffusion" -> R.string.s_348
            "high_noise" -> R.string.s_349
            "uncond" -> R.string.s_350
            "clip_l" -> R.string.draw_tokenizer_vocab
            "clip_g" -> R.string.draw_tokenizer_merges
            "clip_vision" -> R.string.s_353
            "t5xxl" -> R.string.s_354
            "llm" -> R.string.s_355
            "llm_vision" -> R.string.s_356
            "vae" -> R.string.s_357
            "taesd" -> R.string.draw_taesd_label
            else -> R.string.s_358
        }

        /** 某个类型需要哪些槽位（key to 是否必填）。
         *  多文件模型的配套文件各家不一样，这里按家族给模板；用户选了类型就按模板列槽位。
         *  必填只是「加载前的检查」，不阻止用户尝试（可选槽位留空也允许加载）。 */
        fun familySlots(kind: String, family: String): List<Pair<String, Boolean>> = when {
            kind == KIND_SINGLE && family == FAM_SD15 ->
                listOf("model" to true, "clip_l" to true, "clip_g" to true, "vae" to false)
            kind == KIND_SINGLE && family == FAM_SDXL ->
                listOf("model" to true, "clip_l" to true, "clip_g" to true, "vae" to true)
            else -> emptyList()
        }

        /** SharedPreferences 文件名与键（只存 UI 偏好） */
        private const val PREFS_NAME = "ponko"
        private const val KEY_USE_LORA = "useLora"
        private const val MAX_LORA_SLOTS = 4
        private const val LORA_SLOT_NONE = "— none —"

        // 绘图参数持久化（提示词按用户要求不保存，每次重来）
        private const val KEY_W = "drawW"
        private const val KEY_H = "drawH"
        private const val KEY_STEPS = "drawSteps"
        private const val KEY_CFG = "drawCfg"
        private const val KEY_ETA = "drawEta"
        private const val KEY_CLIP_SKIP = "drawClipSkip"
        private const val KEY_SEED = "drawSeed"
        private const val KEY_LORA_SCALE = "drawLoraScale"
        private const val KEY_SAMPLER = "drawSampler"
        private const val KEY_SCHEDULER = "drawScheduler"

        // 图生图参数（与文生图分开存）
        private const val KEY_I2I_W = "i2iW"
        private const val KEY_I2I_H = "i2iH"
        private const val KEY_I2I_STEPS = "i2iSteps"
        private const val KEY_I2I_CFG = "i2iCfg"
        private const val KEY_I2I_ETA = "i2iEta"
        private const val KEY_I2I_CLIP_SKIP = "i2iClipSkip"
        private const val KEY_I2I_SEED = "i2iSeed"
        private const val KEY_I2I_STRENGTH = "i2iStrength"
        private const val KEY_I2I_LORA_SCALE = "i2iLoraScale"
        private const val KEY_I2I_SAMPLER = "i2iSampler"
        private const val KEY_I2I_SCHEDULER = "i2iScheduler"

        private const val KEY_HIRES_ENABLED = "hiresEnabled"
        private const val KEY_HIRES_UPSCALER = "hiresUpscaler"
        private const val KEY_HIRES_MODEL_PATH = "hiresUpscalerModelPath"
        private const val KEY_HIRES_SCALE = "hiresScale"
        private const val KEY_HIRES_TARGET_W = "hiresTargetWidth"
        private const val KEY_HIRES_TARGET_H = "hiresTargetHeight"
        private const val KEY_HIRES_STEPS = "hiresSteps"
        private const val KEY_HIRES_DENOISE = "hiresDenoise"
        private const val KEY_HIRES_TILE = "hiresTileSize"
        private const val KEY_ADETAILER_ENABLED = "drawAdetailerEnabled"
        private const val KEY_ADETAILER_MODEL_PATH = "drawAdetailerModelPath"
        private const val KEY_ADETAILER_PROMPT = "drawAdetailerPrompt"
        private const val KEY_ADETAILER_NEGATIVE_PROMPT = "drawAdetailerNegativePrompt"
        private const val KEY_ADETAILER_CONFIDENCE = "drawAdetailerConfidence"
        private const val KEY_ADETAILER_RESOLUTION = "drawAdetailerResolution"
        private const val KEY_ADETAILER_PADDING = "drawAdetailerPadding"
        private const val KEY_ADETAILER_MASK_BLUR = "drawAdetailerMaskBlur"
        private const val KEY_ADETAILER_DENOISE = "drawAdetailerDenoise"
        private const val KEY_ADETAILER_STEPS = "drawAdetailerSteps"
    }

    private val c: Context get() = act

    /** Hearth inference context handle (0 = not loaded). */
    private var sdHandle: Long = 0L
    private var mainModel: File? = null
    private var vaeModel: File? = null

    /** 当前模型集（单文件 = 只填主模型；多文件 = 主模型 + 各槽位配套文件） */
    private var activeSet: ModelSet? = null
    /** 下一个要导入的类型（由「模型」页的类型菜单设定） */
    var pendingKind: String = KIND_SINGLE
    var pendingFamily: String = FAM_SD15
    /** true = 下一个导入的文件用来开一个新模型集 */
    var pendingNewSet: Boolean = false

    /** 推理线程数：默认 4（多数手机的大核数；开太多会跑到小核上，反而变慢） */
    private var nThreads: Int = prefs().getInt(KEY_CPU_THREADS, 4).coerceIn(
        1, Runtime.getRuntime().availableProcessors().coerceIn(1, 12)
    )

    /** FlashAttention（CLIP + UNet）。实测不开每步慢约 28%（6 步 125s vs 98s）。 */
    private var flashAttn: Boolean = true

    /** 外部（MainActivity）通知：当前已加载语言模型。用于「生成」按钮给出更准确的提示。 */
    var llmLoaded: Boolean = false

    /** 状态回调：把绘图页的进度/结果同步到主界面顶栏。(文本, 是否出错) */
    var onStatus: ((String, Boolean) -> Unit)? = null

    /** 运行方式：true = 尝试 GPU（Vulkan），false = 纯 CPU。由模型页的「运行方式」决定。 */
    var useGpu: Boolean = true

    /** 当前绘图实际用的后端（"CPU" / "Vulkan0"），显示在状态行 */
    private var activeBackend: String = "Vulkan"
    private var activeVulkanDiagnostic: String = ""

    /**
     * 是否启用 LoRA 加速（挂 LCM-LoRA，压低步数）。
     * 状态持久化到 SharedPreferences —— 否则每次启动 App 都要重新勾。
     */
    var useLora: Boolean = c.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        .getBoolean(KEY_USE_LORA, false)
        set(value) {
            field = value
            c.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit().putBoolean(KEY_USE_LORA, value).apply()
        }

    /** 当前选中的 LoRA（null = 自动取目录里第一个） */
    private var loraFile: File? = null

    /** SD 管线就绪时回调（MainActivity 借此切到绘图模式） */
    var onPipelineReady: (() -> Unit)? = null

    /** 是否已就绪（可生成） */
    fun isReady(): Boolean = sdHandle != 0L

    /**
     * 把**已经解析好的出图参数**同步给 [DrawCore]，供 API 的绘图接口使用。
     *
     * 必须在主线程调用：EditText 只能在主线程读，而且 Activity 销毁后控件也不该再被依赖，
     * 所以 DrawCore 存的是「值」而不是控件引用。
     */
    fun syncDrawCore() {
        if (!::stepsEdit.isInitialized || sdHandle == 0L) return
        val loras = selectedLoras("t2i")
        DrawCore.updateParams(
            DrawCore.Params(
                steps = stepsEdit.text.toString().toIntOrNull()?.coerceIn(1, 150) ?: 20,
                cfg = cfgEdit.text.toString().toFloatOrNull()?.coerceIn(1f, 30f) ?: 7.0f,
                width = (widthEdit.text.toString().toIntOrNull() ?: 512)
                    .let { (it / 64) * 64 }.coerceIn(64, 2048),
                height = (heightEdit.text.toString().toIntOrNull() ?: 512)
                    .let { (it / 64) * 64 }.coerceIn(64, 2048),
                negative = negEdit.text.toString(),
                sampler = pickSampler(loras.isNotEmpty()),
                scheduler = pickScheduler(loras.isNotEmpty()),
                loras = loras,
            )
        )
    }

    // 控件
    private lateinit var statusText: TextView
    private lateinit var quantText: TextView
    private lateinit var quantizeUnetButton: Button
    private lateinit var resAdapterButton: Button
    private lateinit var fuseLoraButton: Button
    private var pendingResAdapterSet: ModelSet? = null
    private var pendingResAdapterLoraUri: Uri? = null
    private var pendingResAdapterQuantType: String = "q5_0"
    private lateinit var modelCardTitle: TextView
    private lateinit var promptEdit: EditText
    private lateinit var negEdit: EditText
    private lateinit var stepsEdit: EditText
    private lateinit var cfgEdit: EditText
    private lateinit var etaEdit: EditText
    private lateinit var clipSkipSeekBar: android.widget.SeekBar
    private lateinit var seedEdit: EditText
    private lateinit var loraCheck: CheckBox
    private lateinit var loraHint: TextView
    private lateinit var genBtn: Button

    /** 是否正在生成（同一个按钮在「开始生成 / 中断生成」之间切换） */
    private var generating = false
    private lateinit var progressText: TextView
    private lateinit var progressBar: ProgressBar

    // Width and height are normalised to Hearth's latent-compatible multiples.
    private lateinit var widthEdit: EditText
    private lateinit var heightEdit: EditText

    // 采样器 / 调度器（第 0 项 = 自动）
    private lateinit var samplerSpinner: android.widget.Spinner
    private lateinit var schedulerSpinner: android.widget.Spinner

    /** Four independent LoRA selectors and strengths for text-to-image. */
    private lateinit var loraScaleEdit: EditText
    private val t2iLoraSpinners = mutableListOf<android.widget.Spinner>()
    private val t2iLoraWeights = mutableListOf<EditText>()

    // 顶部菜单：参数 / 结果
    private lateinit var tabParams: TextView
    private lateinit var tabResult: TextView
    private lateinit var paramPane: LinearLayout
    private lateinit var resultPane: LinearLayout
    private lateinit var pager: ViewPager

    // 结果面板
    private lateinit var resultImg: ImageView
    private lateinit var resultInfo: TextView
    private lateinit var saveBtn: Button
    private lateinit var historyRow: LinearLayout
    private lateinit var histEmpty: TextView

    /** 最近一次生成的图（结果页「保存到相册」用） */
    private var lastImage: Bitmap? = null

    // ---- 结果面板子页：结果 / 过程（生成过程实时预览） ----
    private lateinit var tabResResult: TextView
    private lateinit var tabResProcess: TextView
    private lateinit var resultContent: LinearLayout
    private lateinit var processContent: LinearLayout
    private lateinit var previewImg: ImageView
    private lateinit var procStepText: TextView
    private lateinit var procProgressBar: ProgressBar
    /** 结果面板当前子页：0=结果 1=过程 */
    private var resultPage = 0
    /** Keep the newest preview waiting for the UI, rather than dropping it and letting the progress bar run ahead. */
    private data class ProcessPreviewFrame(val step: Int, val bitmap: Bitmap)
    private val processPreviewLock = Any()
    private var pendingProcessPreview: ProcessPreviewFrame? = null
    private var processPreviewPostQueued = false
    @Volatile private var previewFrameSeen = false
    @Volatile private var previewFrameStep = 0

    // ---- 模式：文生图 / 图生图 ----
    private lateinit var tabT2i: TextView
    private lateinit var tabI2i: TextView
    private lateinit var t2iContent: LinearLayout
    private lateinit var i2iContent: LinearLayout

    /** 当前参数页模式：0=文生图 1=图生图 2=HiRes Fix */
    private var mode = 0

    // ---- 图生图专属控件 ----
    private lateinit var i2iPromptEdit: EditText
    private lateinit var i2iNegEdit: EditText
    private lateinit var i2iWidthEdit: EditText
    private lateinit var i2iHeightEdit: EditText
    private lateinit var i2iStepsEdit: EditText
    private lateinit var i2iCfgEdit: EditText
    private lateinit var i2iEtaEdit: EditText
    private lateinit var i2iClipSkipSeekBar: android.widget.SeekBar
    private lateinit var i2iSeedEdit: EditText
    private lateinit var i2iStrengthEdit: EditText
    private lateinit var i2iSamplerSpinner: android.widget.Spinner
    private lateinit var i2iSchedulerSpinner: android.widget.Spinner
    private lateinit var i2iLoraCheck: CheckBox
    private lateinit var i2iLoraScaleEdit: EditText
    private val i2iLoraSpinners = mutableListOf<android.widget.Spinner>()
    private val i2iLoraWeights = mutableListOf<EditText>()
    private lateinit var i2iLoraHint: TextView
    private lateinit var i2iGenBtn: Button
    private lateinit var i2iProgressText: TextView
    private lateinit var i2iProgressBar: ProgressBar
    private lateinit var i2iPreview: ImageView
    private lateinit var i2iPickBtn: Button
    private lateinit var i2iImgHint: TextView

    /** 图生图选中的参考图（仅内存，进程结束即失效） */
    private var i2iBitmap: Bitmap? = null

    // ---- HiRes Fix controls ----
    private lateinit var tabHiRes: TextView
    private lateinit var hiresContent: LinearLayout
    private lateinit var hiresEnabledCheck: CheckBox
    private lateinit var hiresUpscalerSpinner: android.widget.Spinner
    private lateinit var hiresModelButton: Button
    private lateinit var hiresModelStatus: TextView
    private lateinit var hiresScaleEdit: EditText
    private lateinit var hiresTargetWidthEdit: EditText
    private lateinit var hiresTargetHeightEdit: EditText
    private lateinit var hiresStepsEdit: EditText
    private lateinit var hiresDenoiseEdit: EditText
    private lateinit var hiresTileEdit: EditText
    private var hiresModelFile: File? = null

    // ADetailer detector and post-generation inpaint controls.
    private lateinit var adetailerEnabledCheck: CheckBox
    private lateinit var adetailerModelButton: Button
    private lateinit var adetailerModelStatus: TextView
    private lateinit var adetailerPromptEdit: EditText
    private lateinit var adetailerNegativePromptEdit: EditText
    private lateinit var adetailerConfidenceEdit: EditText
    private lateinit var adetailerResolutionEdit: EditText
    private lateinit var adetailerPaddingEdit: EditText
    private lateinit var adetailerMaskBlurEdit: EditText
    private lateinit var adetailerDenoiseEdit: EditText
    private lateinit var adetailerStepsEdit: EditText
    private var adetailerModelFile: File? = null

    private lateinit var paramScroll: ScrollView
    private lateinit var resultScroll: ScrollView
    private lateinit var resultToI2iBtn: Button

    /** 保存图片到相册：MainActivity 注入实现（复用它的 MediaStore 逻辑）。参数：图 + 文件名 */
    var onSaveImage: ((Bitmap, String) -> Unit)? = null

    fun build(): View {
        val root = LinearLayout(c).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(pal.bgAlt)
        }

        // ---- 顶部菜单：参数 / 结果（点击或左右翻页都能切） ----
        val tabBar = LinearLayout(c).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(pal.surface)
            setPadding(dp(16), dp(12), dp(16), 0)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        tabParams = tabItem(c.getString(R.string.s_034))
        tabResult = tabItem(c.getString(R.string.s_203))
        tabParams.setOnClickListener { switchPane(toResult = false) }
        tabResult.setOnClickListener { switchPane(toResult = true) }
        tabBar.addView(tabParams, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        tabBar.addView(tabResult, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(tabBar)

        // ---- 参数面板 ----
        // 内容容器；外面会各自包一层 ScrollView，再交给 ViewPager2 做左右翻页
        paramPane = LinearLayout(c).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(10), dp(16), dp(12))
        }
        // 文生图内容容器：下面的提示词/按钮/参数都先进这里，
        // 再和「图生图」页面一起放进模式容器，由子标签切换。
        t2iContent = LinearLayout(c).apply { orientation = LinearLayout.VERTICAL }

        // ---- 模型状态（模型统一在「模型」页选择并加载） ----
        val modelCard = card()
        modelCardTitle = title(c.getString(R.string.s_164))
        modelCard.addView(modelCardTitle)
        statusText = body(c.getString(R.string.s_115))
        modelCard.addView(statusText)
        quantText = TextView(c).apply {
            textSize = 11.5f
            setTextColor(subText)
            setPadding(0, dp(6), 0, 0)
        }
        modelCard.addView(quantText)
        quantizeUnetButton = Button(c).apply {
            text = c.getString(R.string.draw_quantize_unet)
            visibility = View.GONE
            setOnClickListener { showUnetQuantizer() }
        }
        modelCard.addView(quantizeUnetButton, matchWrap(top = 8))
        resAdapterButton = Button(c).apply {
            text = c.getString(R.string.draw_resadapter_button)
            visibility = View.GONE
            setOnClickListener { showResAdapterWizard() }
        }
        modelCard.addView(resAdapterButton, matchWrap(top = 4))
        fuseLoraButton = Button(c).apply {
            text = c.getString(R.string.draw_lora_fusion_button)
            visibility = View.GONE
            setOnClickListener { showLoraFusionWizard() }
        }
        modelCard.addView(fuseLoraButton, matchWrap(top = 4))
        paramPane.addView(modelCard)

        // ---- 模式子标签：文生图 / 图生图 ----
        val modeBar = LinearLayout(c).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(pal.surface)
            setPadding(dp(8), dp(2), dp(8), dp(2))
        }
        tabT2i = modeTab(c.getString(R.string.s_207))
        tabI2i = modeTab(c.getString(R.string.s_208)).apply { visibility = View.GONE }
        tabHiRes = modeTab(c.getString(R.string.draw_hires_tab)).apply { visibility = View.GONE }
        tabT2i.setOnClickListener { switchMode(0) }
        tabI2i.setOnClickListener { switchMode(1) }
        tabHiRes.setOnClickListener { switchMode(2) }
        modeBar.addView(tabT2i, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        modeBar.addView(tabI2i, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        modeBar.addView(tabHiRes, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        paramPane.addView(modeBar, matchWrap())

        // ---- 提示词 ----
        val promptCard = card()
        promptCard.addView(title(c.getString(R.string.s_108)))
        promptEdit = labeledEdit(c.getString(R.string.s_036), singleLine = false, minLines = 3)
        promptCard.addView(promptEdit, matchWrap(top = 6))
        promptCard.addView(smallLabel(c.getString(R.string.s_177)))
        negEdit = labeledEdit(c.getString(R.string.s_216), singleLine = false, minLines = 2)
        promptCard.addView(negEdit, matchWrap(top = 4))
        t2iContent.addView(promptCard)

        // ---- 生成 / 中断（同一个按钮），放在提示词与参数之间，方便盯着进度 ----
        genBtn = Button(c).apply {
            text = c.getString(R.string.s_098)
            setBackgroundColor(primary)
            setTextColor(Color.WHITE)
            typeface = Typeface.DEFAULT_BOLD
            setOnClickListener { if (generating) doCancel() else generateFromUi() }
        }
        t2iContent.addView(genBtn, matchWrap(top = 2))

        progressText = TextView(c).apply {
            textSize = 12f
            setTextColor(subText)
            visibility = View.GONE
            setPadding(0, dp(8), 0, 0)
        }
        t2iContent.addView(progressText)

        progressBar = ProgressBar(c, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = 0
            visibility = View.GONE
        }
        t2iContent.addView(progressBar, matchWrap(top = 4))

        // ---- 参数 ----
        val paramCard = card()
        paramCard.addView(title(c.getString(R.string.s_065)))

        // Dimensions are restored from the previous generation settings.
        widthEdit = smallNumber(loadParam(KEY_W, defOf("t2i", "w", "512")))
        heightEdit = smallNumber(loadParam(KEY_H, defOf("t2i", "h", "512")))
        paramCard.addView(whRow(c.getString(R.string.s_070),
            c.getString(R.string.s_068), widthEdit, heightEdit))

        // 采样器 / 调度器
        samplerSpinner = choiceSpinner(
            listOf(c.getString(R.string.s_169)) + HearthEngine.Sampler.HEARTH_SUPPORTED.map { it.label }
        )
        schedulerSpinner = choiceSpinner(
            listOf(c.getString(R.string.s_168)) + HearthEngine.Scheduler.HEARTH_SUPPORTED.map { it.label }
        )
        samplerSpinner.setSelection(loadInt(KEY_SAMPLER, defOfInt("t2i", "sampler", 0)).coerceIn(0, samplerSpinner.adapter.count - 1), false)
        schedulerSpinner.setSelection(loadInt(KEY_SCHEDULER, defOfInt("t2i", "scheduler", 0)).coerceIn(0, schedulerSpinner.adapter.count - 1), false)
        syncTcdPresetOnRestore(samplerSpinner, schedulerSpinner, KEY_SAMPLER, KEY_SCHEDULER)
        syncKarrasPresetOnRestore(samplerSpinner, schedulerSpinner, KEY_SCHEDULER)
        // Restore first, then attach listeners so initial values aren't persisted as user changes.
        samplerSpinner.onItemSelectedListener = samplerPresetListener(KEY_SAMPLER, samplerSpinner, schedulerSpinner)
        schedulerSpinner.onItemSelectedListener = schedulerPresetListener(KEY_SCHEDULER, KEY_SAMPLER, samplerSpinner)
        paramCard.addView(paramRow(c.getString(R.string.s_191), samplerSpinner, c.getString(R.string.s_015)))
        paramCard.addView(paramRow(c.getString(R.string.s_176), schedulerSpinner, c.getString(R.string.s_014)))

        stepsEdit = smallNumber(loadParam(KEY_STEPS, defOf("t2i", "steps", builtinSteps())))
        cfgEdit = smallNumber(loadParam(KEY_CFG, defOf("t2i", "cfg", builtinCfg())))
        etaEdit = smallNumber(loadParam(KEY_ETA, defOf("t2i", "eta", "-1")))
        val (clipSkipView, clipSkipBar) = buildClipSkipControl(
            KEY_CLIP_SKIP,
            loadInt(KEY_CLIP_SKIP, defOfInt("t2i", "clipSkip", 1))
        )
        clipSkipSeekBar = clipSkipBar
        seedEdit = smallNumber(loadParam(KEY_SEED, defOf("t2i", "seed", "-1")))
        bindParam(widthEdit, KEY_W)
        bindParam(heightEdit, KEY_H)
        bindParam(stepsEdit, KEY_STEPS)
        bindParam(cfgEdit, KEY_CFG)
        bindParam(etaEdit, KEY_ETA)
        bindParam(seedEdit, KEY_SEED)
        paramCard.addView(paramRow(c.getString(R.string.s_192), stepsEdit, c.getString(R.string.s_179)))
        paramCard.addView(paramRow(c.getString(R.string.s_013), cfgEdit, c.getString(R.string.s_178)))
        paramCard.addView(paramRow(c.getString(R.string.draw_eta_label), etaEdit, c.getString(R.string.draw_eta_hint)))
        paramCard.addView(clipSkipView, matchWrap(top = 6))
        paramCard.addView(paramRow(c.getString(R.string.s_193), seedEdit, c.getString(R.string.s_008)))

        // ---- LoRA 加速开关 ----
        loraCheck = CheckBox(c).apply {
            text = c.getString(R.string.s_022)
            textSize = 13f
            setTextColor(textColor)
            isChecked = useLora
            setPadding(0, dp(10), 0, 0)
            setOnCheckedChangeListener { _, checked ->
                useLora = checked
                // 切换 LoRA = 套用该状态的那套默认值（开/关各一套，可用「设为默认值」覆盖）
                if (checked && activeLora() == null) toast(c.getString(R.string.s_186))
                if (::i2iLoraCheck.isInitialized) i2iLoraCheck.isChecked = checked
                refreshLoraHint()
                if (::i2iContent.isInitialized) {
                    applyDefaults("t2i")
                    applyDefaults("i2i")
                }
            }
        }
        paramCard.addView(loraCheck)
        t2iLoraSpinners.clear()
        t2iLoraWeights.clear()
        for (slot in 0 until MAX_LORA_SLOTS) {
            val key = loraWeightKey("t2i", slot)
            val defaultWeight = if (slot == 0) defOf("t2i", "loraScale", "1.0")
                else defOf("t2i", "loraScale" + (slot + 1), "1.0")
            val weight = smallNumber(loadParam(key, defaultWeight))
            if (slot == 0) loraScaleEdit = weight
            bindParam(weight, key)
            val selector = android.widget.Spinner(c)
            t2iLoraSpinners.add(selector)
            t2iLoraWeights.add(weight)
            paramCard.addView(loraSlotRow(slot, selector, weight), matchWrap(top = 4))
        }
        loraHint = TextView(c).apply {
            textSize = 11f
            setTextColor(subText)
            setPadding(0, dp(2), 0, 0)
        }
        paramCard.addView(loraHint)
        t2iContent.addView(paramCard)
        t2iContent.addView(defaultButtonsRow("t2i"), matchWrap(top = 10))
        // ADetailer will be surfaced when its detector/inpaint path is ported to Hearth.

        // ---- 图生图页面 ----
        i2iContent = buildI2iContent()

        // ---- HiRes Fix 页面 ----
        hiresContent = buildHiresContent()
        refreshLoraHint()

        // 模式容器：用 visibility 切换，不用嵌套 ViewPager，
        // 否则会和外层「参数/结果」的左右翻页抢手势。
        val paramHost = FrameLayout(c)
        paramHost.addView(t2iContent, ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        paramHost.addView(i2iContent, ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        paramHost.addView(hiresContent, ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        paramPane.addView(paramHost, matchWrap())

        // ---- 结果面板 ----
        resultPane = LinearLayout(c).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(10), dp(16), dp(12))
        }

        // ---- 结果面板内的子标签：结果 / 过程（过程页逐步刷新预览）----
        val resBar = LinearLayout(c).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(pal.surface)
            setPadding(dp(8), dp(2), dp(8), dp(2))
        }
        tabResResult = modeTab(c.getString(R.string.s_203))
        tabResProcess = modeTab(c.getString(R.string.s_373))
        tabResResult.setOnClickListener { switchResultPage(0) }
        tabResProcess.setOnClickListener { switchResultPage(1) }
        resBar.addView(tabResResult, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        resBar.addView(tabResProcess, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        resultPane.addView(resBar, matchWrap())

        resultContent = LinearLayout(c).apply { orientation = LinearLayout.VERTICAL }
        resultPane.addView(resultContent, matchWrap())
        processContent = LinearLayout(c).apply { orientation = LinearLayout.VERTICAL }
        resultPane.addView(processContent, matchWrap())

        // 过程页：逐步刷新的预览图 + 步骤进度
        val procCard = card()
        procCard.addView(title(c.getString(R.string.s_373)))
        previewImg = ImageView(c).apply {
            adjustViewBounds = true
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(pal.fieldBg)
            setPadding(dp(4), dp(6), dp(4), dp(6))
            minimumHeight = dp(180)
        }
        procCard.addView(previewImg, matchWrap(top = 6))
        procStepText = TextView(c).apply {
            textSize = 12f
            setTextColor(subText)
            text = c.getString(R.string.s_375)
        }
        procCard.addView(procStepText, matchWrap(top = 8))
        procProgressBar = ProgressBar(c, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = 0
        }
        procCard.addView(procProgressBar, matchWrap(top = 6))
        procCard.addView(TextView(c).apply {
            text = c.getString(R.string.s_374)
            textSize = 11f
            setTextColor(subText)
            setPadding(0, dp(6), 0, 0)
        })
        processContent.addView(procCard)

        val resultCard = card()
        resultCard.addView(title(c.getString(R.string.s_148)))
        resultImg = ImageView(c).apply {
            adjustViewBounds = true
            setPadding(dp(4), dp(10), dp(4), dp(8))
        }
        resultCard.addView(resultImg, matchWrap())
        resultInfo = TextView(c).apply {
            textSize = 12f
            setTextColor(subText)
            text = c.getString(R.string.s_184)
        }
        resultCard.addView(resultInfo)
        saveBtn = Button(c).apply {
            text = c.getString(R.string.s_045)
            setBackgroundColor(primary)
            setTextColor(Color.WHITE)
            typeface = Typeface.DEFAULT_BOLD
            setOnClickListener {
                val b = lastImage
                if (b == null) toast(c.getString(R.string.s_183)) else onSaveImage?.invoke(b, saveName())
            }
        }
        resultCard.addView(saveBtn, matchWrap(top = 10))

        resultToI2iBtn = Button(c).apply {
            // Hearth currently supports text-to-image only; do not advertise a dead img2img path.
            visibility = View.GONE
            text = c.getString(R.string.s_225)
            textSize = 13f
            setTextColor(primary)
            setBackgroundColor(pal.primarySoft)
            setOnClickListener { sendResultToI2i() }
        }
        val sendRow = LinearLayout(c).apply { orientation = LinearLayout.HORIZONTAL }
        sendRow.addView(resultToI2iBtn,
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        resultCard.addView(sendRow, matchWrap(top = 8))
        resultContent.addView(resultCard)

        // ---- 生成历史（只活在内存里，App 进程结束即清空） ----
        val histCard = card()
        histCard.addView(title(c.getString(R.string.s_146)))
        histCard.addView(TextView(c).apply {
            text = c.getString(R.string.s_067)
            textSize = 11f
            setTextColor(subText)
            setPadding(0, dp(4), 0, 0)
        })
        histEmpty = TextView(c).apply {
            text = c.getString(R.string.s_182)
            textSize = 12f
            setTextColor(subText)
            setPadding(0, dp(10), 0, 0)
        }
        histCard.addView(histEmpty)
        historyRow = LinearLayout(c).apply { orientation = LinearLayout.HORIZONTAL }
        val hsv = android.widget.HorizontalScrollView(c).apply {
            isHorizontalScrollBarEnabled = false
            addView(historyRow)
        }
        histCard.addView(hsv, matchWrap(top = 8))
        val clearHistBtn = Button(c).apply {
            text = c.getString(R.string.s_138)
            textSize = 12f
            setTextColor(subText)
            setOnClickListener {
                if (DrawHistory.size() == 0) {
                    toast(c.getString(R.string.s_064))
                } else {
                    android.app.AlertDialog.Builder(act)
                        .setMessage(c.getString(R.string.s_137))
                        .setPositiveButton(c.getString(R.string.s_136)) { _, _ -> DrawHistory.clear(); refreshHistory() }
                        .setNegativeButton(c.getString(R.string.s_066), null)
                        .show()
                }
            }
        }
        histCard.addView(clearHistBtn, matchWrap(top = 10))
        resultContent.addView(histCard)
        refreshHistory()

        // ---- 参数 / 结果：ViewPager 跟手翻页（同手机桌面） ----
        // 用老版 ViewPager 而不是 ViewPager2：它的 PagerAdapter 原生支持直接复用
        // 已有的 View；ViewPager2 内部是 RecyclerView，重复 attach 同一个 View 会崩。
        pager = ViewPager(c).apply {
            paramScroll = scrollWrap(paramPane)
            resultScroll = scrollWrap(resultPane)
            adapter = PaneAdapter(listOf(paramScroll, resultScroll))
            addOnPageChangeListener(object : ViewPager.SimpleOnPageChangeListener() {
                override fun onPageSelected(position: Int) {
                    highlightTabs(toResult = position == 1)
                }
            })
        }
        root.addView(pager, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))

        refreshLoraHint()
        refreshQuantText()
        switchMode(0)
        switchPane(toResult = false)
        switchResultPage(0)
        return root
    }

    // ================= 图生图页面 =================

    /** 构建「图生图」参数页（独立的一套控件，持久化键也独立） */
    private fun buildI2iContent(): LinearLayout {
        val box = LinearLayout(c).apply { orientation = LinearLayout.VERTICAL }

        // ---- 参考图 ----
        val imgCard = card()
        imgCard.addView(title(c.getString(R.string.s_209)))
        i2iPreview = ImageView(c).apply {
            adjustViewBounds = true
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(pal.fieldBg)
            visibility = View.GONE
            setPadding(dp(4), dp(4), dp(4), dp(4))
        }
        imgCard.addView(i2iPreview, matchWrap(top = 8))
        i2iPickBtn = Button(c).apply {
            text = c.getString(R.string.s_210)
            setBackgroundColor(primary)
            setTextColor(Color.WHITE)
            typeface = Typeface.DEFAULT_BOLD
            setOnClickListener { pickImage() }
        }
        imgCard.addView(i2iPickBtn, matchWrap(top = 10))
        i2iImgHint = TextView(c).apply {
            text = c.getString(R.string.s_211)
            textSize = 11.5f
            setTextColor(subText)
            setPadding(0, dp(6), 0, 0)
        }
        imgCard.addView(i2iImgHint)
        box.addView(imgCard)

        // ---- 提示词 ----
        val promptCard = card()
        promptCard.addView(title(c.getString(R.string.s_108)))
        i2iPromptEdit = labeledEdit(c.getString(R.string.s_036), singleLine = false, minLines = 3)
        promptCard.addView(i2iPromptEdit, matchWrap(top = 6))
        promptCard.addView(smallLabel(c.getString(R.string.s_177)))
        i2iNegEdit = labeledEdit(c.getString(R.string.s_216), singleLine = false, minLines = 2)
        promptCard.addView(i2iNegEdit, matchWrap(top = 4))
        box.addView(promptCard)

        // ---- 生成 / 中断 ----
        i2iGenBtn = Button(c).apply {
            text = c.getString(R.string.s_098)
            setBackgroundColor(primary)
            setTextColor(Color.WHITE)
            typeface = Typeface.DEFAULT_BOLD
            setOnClickListener { if (generating) doCancel() else generateFromUiI2i() }
        }
        box.addView(i2iGenBtn, matchWrap(top = 2))

        i2iProgressText = TextView(c).apply {
            textSize = 12f
            setTextColor(subText)
            visibility = View.GONE
            setPadding(0, dp(8), 0, 0)
        }
        box.addView(i2iProgressText)
        i2iProgressBar = ProgressBar(c, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = 0
            visibility = View.GONE
        }
        box.addView(i2iProgressBar, matchWrap(top = 4))

        // ---- 参数 ----
        val paramCard = card()
        paramCard.addView(title(c.getString(R.string.s_065)))

        // 重绘强度：越小越接近原图
        i2iStrengthEdit = smallNumber(loadParam(KEY_I2I_STRENGTH, defOf("i2i", "strength", "0.75")))
        bindParam(i2iStrengthEdit, KEY_I2I_STRENGTH)
        paramCard.addView(paramRow(c.getString(R.string.s_212), i2iStrengthEdit, c.getString(R.string.s_213)))

        // 尺寸：默认选图时会同步成参考图尺寸（64 倍数）
        i2iWidthEdit = smallNumber(loadParam(KEY_I2I_W, defOf("i2i", "w", "512")))
        i2iHeightEdit = smallNumber(loadParam(KEY_I2I_H, defOf("i2i", "h", "512")))
        bindParam(i2iWidthEdit, KEY_I2I_W)
        bindParam(i2iHeightEdit, KEY_I2I_H)
        paramCard.addView(whRow(c.getString(R.string.s_070), c.getString(R.string.s_068), i2iWidthEdit, i2iHeightEdit))

        i2iSamplerSpinner = choiceSpinner(
            listOf(c.getString(R.string.s_169)) + HearthEngine.Sampler.HEARTH_SUPPORTED.map { it.label }
        )
        i2iSchedulerSpinner = choiceSpinner(
            listOf(c.getString(R.string.s_168)) + HearthEngine.Scheduler.HEARTH_SUPPORTED.map { it.label }
        )
        i2iSamplerSpinner.setSelection(loadInt(KEY_I2I_SAMPLER, defOfInt("i2i", "sampler", 0)).coerceIn(0, i2iSamplerSpinner.adapter.count - 1), false)
        i2iSchedulerSpinner.setSelection(loadInt(KEY_I2I_SCHEDULER, defOfInt("i2i", "scheduler", 0)).coerceIn(0, i2iSchedulerSpinner.adapter.count - 1), false)
        syncTcdPresetOnRestore(i2iSamplerSpinner, i2iSchedulerSpinner, KEY_I2I_SAMPLER, KEY_I2I_SCHEDULER)
        syncKarrasPresetOnRestore(i2iSamplerSpinner, i2iSchedulerSpinner, KEY_I2I_SCHEDULER)
        i2iSamplerSpinner.onItemSelectedListener = samplerPresetListener(KEY_I2I_SAMPLER, i2iSamplerSpinner, i2iSchedulerSpinner)
        i2iSchedulerSpinner.onItemSelectedListener = schedulerPresetListener(KEY_I2I_SCHEDULER, KEY_I2I_SAMPLER, i2iSamplerSpinner)
        paramCard.addView(paramRow(c.getString(R.string.s_191), i2iSamplerSpinner, c.getString(R.string.s_015)))
        paramCard.addView(paramRow(c.getString(R.string.s_176), i2iSchedulerSpinner, c.getString(R.string.s_014)))

        i2iStepsEdit = smallNumber(loadParam(KEY_I2I_STEPS, defOf("i2i", "steps", builtinSteps())))
        i2iCfgEdit = smallNumber(loadParam(KEY_I2I_CFG, defOf("i2i", "cfg", builtinCfg())))
        i2iEtaEdit = smallNumber(loadParam(KEY_I2I_ETA, defOf("i2i", "eta", "-1")))
        val (i2iClipSkipView, i2iClipSkipBar) = buildClipSkipControl(
            KEY_I2I_CLIP_SKIP,
            loadInt(KEY_I2I_CLIP_SKIP, defOfInt("i2i", "clipSkip", 1))
        )
        i2iClipSkipSeekBar = i2iClipSkipBar
        i2iSeedEdit = smallNumber(loadParam(KEY_I2I_SEED, defOf("i2i", "seed", "-1")))
        bindParam(i2iStepsEdit, KEY_I2I_STEPS)
        bindParam(i2iCfgEdit, KEY_I2I_CFG)
        bindParam(i2iEtaEdit, KEY_I2I_ETA)
        bindParam(i2iSeedEdit, KEY_I2I_SEED)
        paramCard.addView(paramRow(c.getString(R.string.s_192), i2iStepsEdit, c.getString(R.string.s_179)))
        paramCard.addView(paramRow(c.getString(R.string.s_013), i2iCfgEdit, c.getString(R.string.s_178)))
        paramCard.addView(paramRow(c.getString(R.string.draw_eta_label), i2iEtaEdit, c.getString(R.string.draw_eta_hint)))
        paramCard.addView(i2iClipSkipView, matchWrap(top = 6))
        paramCard.addView(paramRow(c.getString(R.string.s_193), i2iSeedEdit, c.getString(R.string.s_008)))

        // LoRA（与文生图共用同一个开关，两边 checkbox 相互同步）
        i2iLoraCheck = CheckBox(c).apply {
            text = c.getString(R.string.s_022)
            textSize = 13f
            setTextColor(textColor)
            isChecked = useLora
            setPadding(0, dp(10), 0, 0)
            setOnCheckedChangeListener { _, checked ->
                useLora = checked
                if (checked && activeLora() == null) toast(c.getString(R.string.s_186))
                if (::loraCheck.isInitialized) loraCheck.isChecked = checked
                refreshLoraHint()
                if (::t2iContent.isInitialized) {
                    applyDefaults("t2i")
                    applyDefaults("i2i")
                }
            }
        }
        paramCard.addView(i2iLoraCheck)
        i2iLoraSpinners.clear()
        i2iLoraWeights.clear()
        for (slot in 0 until MAX_LORA_SLOTS) {
            val key = loraWeightKey("i2i", slot)
            val defaultWeight = if (slot == 0) defOf("i2i", "loraScale", "1.0")
                else defOf("i2i", "loraScale" + (slot + 1), "1.0")
            val weight = smallNumber(loadParam(key, defaultWeight))
            if (slot == 0) i2iLoraScaleEdit = weight
            bindParam(weight, key)
            val selector = android.widget.Spinner(c)
            i2iLoraSpinners.add(selector)
            i2iLoraWeights.add(weight)
            paramCard.addView(loraSlotRow(slot, selector, weight), matchWrap(top = 4))
        }
        i2iLoraHint = TextView(c).apply {
            textSize = 11f
            setTextColor(subText)
            setPadding(0, dp(2), 0, 0)
        }
        paramCard.addView(i2iLoraHint)
        paramCard.addView(defaultButtonsRow("i2i"), matchWrap(top = 10))
        box.addView(paramCard)

        return box
    }

    /** 切换参数页：0=文生图 1=图生图 2=HiRes Fix */
    private fun switchMode(m: Int) {
        if (!::t2iContent.isInitialized) return
        // The Hearth adapter currently supports text-to-image only.
        mode = 0
        t2iContent.visibility = View.VISIBLE
        if (::i2iContent.isInitialized) i2iContent.visibility = View.GONE
        if (::hiresContent.isInitialized) hiresContent.visibility = View.GONE
        tabT2i.setTextColor(primary)
        tabT2i.typeface = Typeface.DEFAULT_BOLD
        tabT2i.setBackgroundColor(pal.primarySoft)
        refreshModelCard()
    }

    /** 模式子标签样式（比顶部页签略小） */
    private fun modeTab(t: String) = TextView(c).apply {
        text = t
        textSize = 13f
        gravity = Gravity.CENTER
        setPadding(0, dp(9), 0, dp(9))
    }

    // ================= ADetailer =================
    private fun buildADetailerContent(): LinearLayout {
        val box = LinearLayout(c).apply { orientation = LinearLayout.VERTICAL }
        val settingsCard = card()
        settingsCard.addView(title(c.getString(R.string.draw_adetailer_title)))
        adetailerEnabledCheck = CheckBox(c).apply {
            text = c.getString(R.string.draw_adetailer_enable)
            textSize = 13f
            setTextColor(textColor)
            isChecked = prefs().getBoolean(KEY_ADETAILER_ENABLED, false)
            setOnCheckedChangeListener { _, checked -> prefs().edit().putBoolean(KEY_ADETAILER_ENABLED, checked).apply() }
        }
        settingsCard.addView(adetailerEnabledCheck, matchWrap(top = 6))
        settingsCard.addView(body(c.getString(R.string.draw_adetailer_hint)))
        adetailerModelFile = prefs().getString(KEY_ADETAILER_MODEL_PATH, null)?.let { File(it) }?.takeIf { it.isFile }
        adetailerModelButton = Button(c).apply {
            text = c.getString(R.string.draw_adetailer_pick_model)
            setBackgroundColor(pal.primarySoft); setTextColor(primary)
            setOnClickListener { pickADetailerModel() }
        }
        settingsCard.addView(adetailerModelButton, matchWrap(top = 8))
        adetailerModelStatus = TextView(c).apply { textSize = 11.5f; setTextColor(subText); setPadding(0, dp(5), 0, 0) }
        settingsCard.addView(adetailerModelStatus)

        adetailerPromptEdit = labeledEdit(c.getString(R.string.draw_adetailer_prompt_hint), true)
            .apply { setText(loadParam(KEY_ADETAILER_PROMPT, "")) }
        bindParam(adetailerPromptEdit, KEY_ADETAILER_PROMPT)
        bindOptionalParam(adetailerPromptEdit, KEY_ADETAILER_PROMPT)
        settingsCard.addView(paramRow(c.getString(R.string.draw_adetailer_prompt), adetailerPromptEdit,
            c.getString(R.string.draw_adetailer_prompt_hint)), matchWrap(top = 4))

        adetailerNegativePromptEdit = labeledEdit(c.getString(R.string.draw_adetailer_negative_prompt_hint), true)
            .apply { setText(loadParam(KEY_ADETAILER_NEGATIVE_PROMPT, "")) }
        bindParam(adetailerNegativePromptEdit, KEY_ADETAILER_NEGATIVE_PROMPT)
        bindOptionalParam(adetailerNegativePromptEdit, KEY_ADETAILER_NEGATIVE_PROMPT)
        settingsCard.addView(paramRow(c.getString(R.string.draw_adetailer_negative_prompt), adetailerNegativePromptEdit,
            c.getString(R.string.draw_adetailer_negative_prompt_hint)), matchWrap(top = 4))

        adetailerConfidenceEdit = smallNumber(loadParam(KEY_ADETAILER_CONFIDENCE, "0.3"))
        bindParam(adetailerConfidenceEdit, KEY_ADETAILER_CONFIDENCE)
        settingsCard.addView(paramRow(c.getString(R.string.draw_adetailer_confidence), adetailerConfidenceEdit, "0.01–0.99"), matchWrap(top = 4))
        adetailerResolutionEdit = smallNumber(loadParam(KEY_ADETAILER_RESOLUTION, "512"))
        bindParam(adetailerResolutionEdit, KEY_ADETAILER_RESOLUTION)
        settingsCard.addView(paramRow(c.getString(R.string.draw_adetailer_resolution), adetailerResolutionEdit,
            c.getString(R.string.draw_adetailer_resolution_hint)), matchWrap(top = 4))
        adetailerPaddingEdit = smallNumber(loadParam(KEY_ADETAILER_PADDING, "32"))
        bindParam(adetailerPaddingEdit, KEY_ADETAILER_PADDING)
        settingsCard.addView(paramRow(c.getString(R.string.draw_adetailer_padding), adetailerPaddingEdit, "0–512 px around the detected region"), matchWrap(top = 4))
        adetailerMaskBlurEdit = smallNumber(loadParam(KEY_ADETAILER_MASK_BLUR, "4"))
        bindParam(adetailerMaskBlurEdit, KEY_ADETAILER_MASK_BLUR)
        settingsCard.addView(paramRow(c.getString(R.string.draw_adetailer_mask_blur), adetailerMaskBlurEdit, "0 disables mask feathering"), matchWrap(top = 4))
        adetailerDenoiseEdit = smallNumber(loadParam(KEY_ADETAILER_DENOISE, "0.4"))
        bindParam(adetailerDenoiseEdit, KEY_ADETAILER_DENOISE)
        settingsCard.addView(paramRow(c.getString(R.string.draw_adetailer_denoise), adetailerDenoiseEdit,
            "0.01–1.0; lower values preserve the original crop"), matchWrap(top = 4))
        adetailerStepsEdit = smallNumber(loadParam(KEY_ADETAILER_STEPS, "8"))
        bindParam(adetailerStepsEdit, KEY_ADETAILER_STEPS)
        settingsCard.addView(paramRow(c.getString(R.string.draw_adetailer_steps), adetailerStepsEdit,
            c.getString(R.string.draw_adetailer_steps_hint)), matchWrap(top = 4))
        box.addView(settingsCard)
        updateADetailerModelStatus()
        return box
    }

    private fun pickADetailerModel() {
        val intent = android.content.Intent(android.content.Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(android.content.Intent.CATEGORY_OPENABLE)
            type = "*/*"
            putExtra(android.content.Intent.EXTRA_TITLE, c.getString(R.string.draw_adetailer_pick_model))
        }
        act.startActivityForResult(intent, REQ_ADETAILER_MODEL)
    }

    private fun updateADetailerModelStatus() {
        if (!::adetailerModelStatus.isInitialized) return
        val model = adetailerModelFile?.takeIf { it.isFile }
        adetailerModelStatus.text = if (model != null)
            c.getString(R.string.draw_adetailer_model_selected, model.name)
        else c.getString(R.string.draw_adetailer_model_none)
    }

    private fun handleADetailerPickedFile(requestCode: Int, resultCode: Int, data: android.content.Intent?): Boolean {
        if (requestCode != REQ_ADETAILER_MODEL) return false
        if (resultCode != Activity.RESULT_OK) return true
        val uri = data?.data ?: return true
        val name = queryDisplayName(uri).orEmpty()
        if (!name.endsWith(".safetensors", ignoreCase = true)) {
            toast(c.getString(R.string.draw_adetailer_bad_model)); return true
        }
        scope.launch {
            try {
                val destination = withContext(Dispatchers.IO) {
                    val dir = File(c.filesDir, ADETAILER_MODEL_DIR).apply { mkdirs() }
                    val safeName = File(name).name.replace(Regex("[^A-Za-z0-9._ -]"), "_").take(120)
                        .ifBlank { "yolov8-detector.safetensors" }
                    val stem = safeName.substringBeforeLast('.', safeName)
                    val ext = safeName.substringAfterLast('.', "")
                    var target = File(dir, safeName)
                    var suffix = 2
                    while (target.exists()) {
                        target = File(dir, "$stem-$suffix" + if (ext.isEmpty()) "" else ".$ext"); suffix++
                    }
                    copyContentUri(uri, target); target
                }
                adetailerModelFile = destination
                prefs().edit().putString(KEY_ADETAILER_MODEL_PATH, destination.absolutePath).apply()
                updateADetailerModelStatus()
                toast(c.getString(R.string.draw_adetailer_imported, destination.name))
            } catch (e: Throwable) {
                toast(c.getString(R.string.draw_adetailer_import_failed, e.message ?: e.javaClass.simpleName))
            }
        }
        return true
    }

    private fun currentADetailerSettings(): HearthEngine.ADetailerSettings = HearthEngine.ADetailerSettings()

    // ================= HiRes Fix 页面 =================
    private fun buildHiresContent(): LinearLayout {
        val box = LinearLayout(c).apply { orientation = LinearLayout.VERTICAL }
        val enableCard = card()
        enableCard.addView(title(c.getString(R.string.draw_hires_title)))
        hiresEnabledCheck = CheckBox(c).apply {
            text = c.getString(R.string.draw_hires_enable)
            textSize = 13f
            setTextColor(textColor)
            isChecked = prefs().getBoolean(KEY_HIRES_ENABLED, false)
            setOnCheckedChangeListener { _, checked -> prefs().edit().putBoolean(KEY_HIRES_ENABLED, checked).apply() }
        }
        enableCard.addView(hiresEnabledCheck, matchWrap(top = 6))
        enableCard.addView(body(c.getString(R.string.draw_hires_enable_hint)))
        box.addView(enableCard)

        val upscalerCard = card()
        upscalerCard.addView(title(c.getString(R.string.draw_hires_upscaler)))
        hiresUpscalerSpinner = choiceSpinner(HearthEngine.HiresUpscaler.entries.map { it.label })
        hiresUpscalerSpinner.setSelection(loadInt(KEY_HIRES_UPSCALER, HearthEngine.HiresUpscaler.LATENT.code)
            .coerceIn(0, HearthEngine.HiresUpscaler.entries.lastIndex), false)
        hiresUpscalerSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: android.widget.AdapterView<*>?, v: View?, pos: Int, id: Long) {
                prefs().edit().putInt(KEY_HIRES_UPSCALER, pos).apply()
                updateHiresModelControls()
            }
            override fun onNothingSelected(p: android.widget.AdapterView<*>?) {}
        }
        upscalerCard.addView(hiresUpscalerSpinner, matchWrap(top = 4))
        upscalerCard.addView(body(c.getString(R.string.draw_hires_upscaler_hint)))
        hiresModelFile = prefs().getString(KEY_HIRES_MODEL_PATH, null)?.let { File(it) }?.takeIf { it.isFile }
        hiresModelButton = Button(c).apply {
            text = c.getString(R.string.draw_hires_pick_model)
            setBackgroundColor(pal.primarySoft)
            setTextColor(primary)
            setOnClickListener { pickHiresUpscalerModel() }
        }
        upscalerCard.addView(hiresModelButton, matchWrap(top = 8))
        hiresModelStatus = TextView(c).apply { textSize = 11.5f; setTextColor(subText); setPadding(0, dp(5), 0, 0) }
        upscalerCard.addView(hiresModelStatus)
        box.addView(upscalerCard)

        val paramsCard = card()
        paramsCard.addView(title(c.getString(R.string.draw_hires_params_title)))
        hiresScaleEdit = smallNumber(loadParam(KEY_HIRES_SCALE, "2.0"))
        bindParam(hiresScaleEdit, KEY_HIRES_SCALE)
        paramsCard.addView(paramRow(c.getString(R.string.draw_hires_scale), hiresScaleEdit, c.getString(R.string.draw_hires_scale_hint)))
        hiresTargetWidthEdit = smallNumber(loadParam(KEY_HIRES_TARGET_W, "0"))
        bindParam(hiresTargetWidthEdit, KEY_HIRES_TARGET_W)
        paramsCard.addView(paramRow(c.getString(R.string.draw_hires_target_width), hiresTargetWidthEdit, "0 = use scale"))
        hiresTargetHeightEdit = smallNumber(loadParam(KEY_HIRES_TARGET_H, "0"))
        bindParam(hiresTargetHeightEdit, KEY_HIRES_TARGET_H)
        paramsCard.addView(paramRow(c.getString(R.string.draw_hires_target_height), hiresTargetHeightEdit, "0 = use scale"))
        hiresStepsEdit = smallNumber(loadParam(KEY_HIRES_STEPS, "0"))
        bindParam(hiresStepsEdit, KEY_HIRES_STEPS)
        paramsCard.addView(paramRow(c.getString(R.string.draw_hires_steps), hiresStepsEdit, c.getString(R.string.draw_hires_steps_hint)))
        hiresDenoiseEdit = smallNumber(loadParam(KEY_HIRES_DENOISE, "0.7"))
        bindParam(hiresDenoiseEdit, KEY_HIRES_DENOISE)
        paramsCard.addView(paramRow(c.getString(R.string.draw_hires_denoise), hiresDenoiseEdit, c.getString(R.string.draw_hires_denoise_hint)))
        hiresTileEdit = smallNumber(loadParam(KEY_HIRES_TILE, "128"))
        bindParam(hiresTileEdit, KEY_HIRES_TILE)
        paramsCard.addView(paramRow(c.getString(R.string.draw_hires_tile), hiresTileEdit, c.getString(R.string.draw_hires_tile_hint)))
        box.addView(paramsCard)
        updateHiresModelControls()
        return box
    }

    private fun updateHiresModelControls() {
        if (!::hiresUpscalerSpinner.isInitialized || !::hiresModelButton.isInitialized || !::hiresModelStatus.isInitialized) return
        val selected = HearthEngine.HiresUpscaler.entries.getOrNull(hiresUpscalerSpinner.selectedItemPosition)
            ?: HearthEngine.HiresUpscaler.LATENT
        val isModel = selected == HearthEngine.HiresUpscaler.MODEL
        hiresModelButton.visibility = if (isModel) View.VISIBLE else View.GONE
        hiresModelStatus.visibility = if (isModel) View.VISIBLE else View.GONE
        val model = hiresModelFile?.takeIf { it.isFile }
        hiresModelStatus.text = if (model != null) c.getString(R.string.draw_hires_model_selected, model.name)
            else c.getString(R.string.draw_hires_model_none)
    }

    private fun pickHiresUpscalerModel() {
        val intent = android.content.Intent(android.content.Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(android.content.Intent.CATEGORY_OPENABLE)
            type = "*/*"
        }
        act.startActivityForResult(intent, REQ_HIRES_MODEL)
    }

    private fun handleHiresUpscalerPickedFile(requestCode: Int, resultCode: Int, data: android.content.Intent?): Boolean {
        if (requestCode != REQ_HIRES_MODEL) return false
        if (resultCode != Activity.RESULT_OK) return true
        val uri = data?.data ?: return true
        val name = queryDisplayName(uri).orEmpty()
        if (name.substringAfterLast('.', "").lowercase() !in setOf("safetensors", "gguf", "pt", "pth", "ckpt")) {
            toast(c.getString(R.string.draw_hires_model_bad_file))
            return true
        }
        scope.launch {
            try {
                val destination = withContext(Dispatchers.IO) {
                    val dir = File(c.filesDir, HIRES_UPSCALER_DIR).apply { mkdirs() }
                    val safeName = File(name).name.replace(Regex("[^A-Za-z0-9._ -]"), "_").take(120)
                        .ifBlank { "upscaler.safetensors" }
                    val stem = safeName.substringBeforeLast('.', safeName)
                    val ext = safeName.substringAfterLast('.', "")
                    var target = File(dir, safeName)
                    var suffix = 2
                    while (target.exists()) {
                        target = File(dir, "$stem-$suffix" + if (ext.isEmpty()) "" else ".$ext")
                        suffix++
                    }
                    copyContentUri(uri, target)
                    target
                }
                hiresModelFile = destination
                prefs().edit().putString(KEY_HIRES_MODEL_PATH, destination.absolutePath).apply()
                updateHiresModelControls()
                toast(c.getString(R.string.draw_hires_model_imported, destination.name))
            } catch (e: Throwable) {
                toast(c.getString(R.string.draw_hires_model_copy_failed, e.message ?: e.javaClass.simpleName))
            }
        }
        return true
    }

    private fun currentHiresSettings(): HearthEngine.HiResSettings = HearthEngine.HiResSettings()

    private fun sendResultToI2i() {
        toast(c.getString(R.string.draw_hearth_i2i_unavailable))
    }

    fun sendImageToI2i(@Suppress("UNUSED_PARAMETER") b: Bitmap) {
        // Callers must not be told an image was queued when the img2img page is intentionally hidden.
        toast(c.getString(R.string.draw_hearth_i2i_unavailable))
    }

    /** 从相册/文件选择参考图 */
    private fun pickImage() {
        val i = android.content.Intent(android.content.Intent.ACTION_GET_CONTENT).apply {
            addCategory(android.content.Intent.CATEGORY_OPENABLE)
            type = "image/*"
        }
        act.startActivityForResult(i, REQ_IMAGE)
    }

    /** 解码选中的图片；超过 2048 像素则按比例下采样，避免 OOM */
    private fun decodePickedImage(uri: Uri): Bitmap? = runCatching {
        val bounds = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
        c.contentResolver.openInputStream(uri)?.use {
            android.graphics.BitmapFactory.decodeStream(it, null, bounds)
        }
        var sample = 1
        val maxSide = maxOf(bounds.outWidth, bounds.outHeight)
        while (maxSide / sample > 2048) sample *= 2
        val opts = android.graphics.BitmapFactory.Options().apply { inSampleSize = sample }
        c.contentResolver.openInputStream(uri)?.use {
            android.graphics.BitmapFactory.decodeStream(it, null, opts)
        }
    }.getOrNull()

    /** 选定参考图后的处理：预览 + 同步输出尺寸 */
    private fun applyPickedImage(bmp: Bitmap) {
        i2iBitmap = bmp
        i2iPreview.setImageBitmap(bmp)
        i2iPreview.visibility = View.VISIBLE
        val w = ((bmp.width / 64) * 64).coerceIn(64, 2048)
        val h = ((bmp.height / 64) * 64).coerceIn(64, 2048)
        i2iWidthEdit.setText(w.toString())
        i2iHeightEdit.setText(h.toString())
        i2iImgHint.text = c.getString(R.string.v_061, bmp.width, bmp.height)
    }

    private fun refreshI2iLoraHint() {
        updateI2iLoraHintText()
    }

    // ================= LoRA =================

    private fun loraWeightKey(page: String, slot: Int): String {
        val base = if (page == "i2i") KEY_I2I_LORA_SCALE else KEY_LORA_SCALE
        return if (slot == 0) base else base + "_slot" + (slot + 1)
    }

    private fun loraSlotKey(page: String, slot: Int): String =
        "drawLora_" + page + "_slot" + (slot + 1)

    private fun loraSlotRow(
        slot: Int,
        selector: android.widget.Spinner,
        weight: EditText,
    ): LinearLayout = LinearLayout(c).apply {
        orientation = LinearLayout.HORIZONTAL
        val label = TextView(c).apply {
            text = "L" + (slot + 1)
            textSize = 12f
            setTextColor(subText)
            gravity = Gravity.CENTER_VERTICAL
        }
        addView(label, LinearLayout.LayoutParams(dp(30), dp(46)))
        addView(selector, LinearLayout.LayoutParams(0, dp(46), 1f))
        weight.hint = "1.0"
        addView(weight, LinearLayout.LayoutParams(dp(64), dp(46)).apply {
            leftMargin = dp(6)
        })
    }

    /** Refresh four per-mode selectors after import, delete, or model selection changes. */
    private fun refreshLoraSelectors() {
        val files = listLoras()
        val names = listOf(LORA_SLOT_NONE) + files.map { it.name }

        fun refresh(spinners: List<android.widget.Spinner>, page: String) {
            spinners.forEachIndexed { slot, spinner ->
                val key = loraSlotKey(page, slot)
                val saved = if (prefs().contains(key)) {
                    prefs().getString(key, "") ?: ""
                } else if (slot == 0) {
                    activeLora()?.name.orEmpty()
                } else {
                    ""
                }
                val adapter = android.widget.ArrayAdapter(
                    c, android.R.layout.simple_spinner_item, names
                ).apply {
                    setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
                }
                spinner.onItemSelectedListener = null
                spinner.adapter = adapter
                spinner.setSelection(names.indexOf(saved).coerceAtLeast(0), false)
                spinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
                    override fun onItemSelected(
                        parent: android.widget.AdapterView<*>?,
                        view: View?,
                        position: Int,
                        id: Long,
                    ) {
                        val chosen = names.getOrNull(position)
                            ?.takeUnless { it == LORA_SLOT_NONE }.orEmpty()
                        prefs().edit().putString(key, chosen).apply()
                        if (page == "t2i") {
                            updateLoraHintText()
                            syncDrawCore()
                        } else {
                            updateI2iLoraHintText()
                        }
                    }

                    override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
                }
            }
        }

        refresh(t2iLoraSpinners, "t2i")
        refresh(i2iLoraSpinners, "i2i")
    }

    /** Selected LoRAs with independent multipliers, bounded to four unique files. */
    private fun selectedLoras(page: String): List<HearthEngine.Lora> {
        if (!useLora) return emptyList()
        val spinners = if (page == "i2i") i2iLoraSpinners else t2iLoraSpinners
        val weights = if (page == "i2i") i2iLoraWeights else t2iLoraWeights
        val files = listLoras()
        val seen = HashSet<String>()
        return spinners.mapIndexedNotNull { slot, spinner ->
            val name = spinner.selectedItem as? String ?: return@mapIndexedNotNull null
            if (name == LORA_SLOT_NONE) return@mapIndexedNotNull null
            val file = files.firstOrNull { it.name == name && it.isFile }
                ?: return@mapIndexedNotNull null
            if (!seen.add(file.absolutePath)) return@mapIndexedNotNull null
            val scale = weights.getOrNull(slot)?.text?.toString()
                ?.toFloatOrNull()?.coerceIn(0f, 2f) ?: 1.0f
            if (scale <= 0f) null else HearthEngine.Lora(file.absolutePath, scale)
        }
    }

    private fun selectedLoraNames(spinners: List<android.widget.Spinner>): List<String> =
        spinners.mapNotNull { it.selectedItem as? String }
            .filter { it != LORA_SLOT_NONE }
            .distinct()

    private fun updateLoraHintText() {
        if (!::loraHint.isInitialized) return
        val names = selectedLoraNames(t2iLoraSpinners).map { File(it).nameWithoutExtension }
        loraHint.text = when {
            names.isEmpty() -> c.getString(R.string.s_118)
            useLora -> c.getString(R.string.v_028, names.joinToString(", "))
            else -> c.getString(R.string.v_029, names.joinToString(", "))
        }
    }

    private fun updateI2iLoraHintText() {
        if (!::i2iLoraHint.isInitialized) return
        val names = selectedLoraNames(i2iLoraSpinners).map { File(it).nameWithoutExtension }
        i2iLoraHint.text = when {
            names.isEmpty() -> c.getString(R.string.s_118)
            useLora -> c.getString(R.string.v_028, names.joinToString(", "))
            else -> c.getString(R.string.v_029, names.joinToString(", "))
        }
    }

    /** If a slot has never been configured or is empty, use the newly imported LoRA as slot 1. */
    private fun selectNewLoraIfFirstSlotEmpty(file: File) {
        val e = prefs().edit().putString("selLora", file.name)
        for (page in listOf("t2i", "i2i")) {
            val key = loraSlotKey(page, 0)
            val saved = prefs().getString(key, null)
            if (saved.isNullOrEmpty()) e.putString(key, file.name)
        }
        e.apply()
    }

    /** LoRA 目录：filesDir/draw/lora */
    private fun loraDir(): File = File(File(c.filesDir, DIR_NAME), "lora").apply { mkdirs() }

    /** 已下载的 LoRA 列表（.safetensors） */
    fun listLoras(): List<File> =
        loraDir().listFiles { f -> f.isFile && f.name.endsWith(".safetensors", ignoreCase = true) }
            ?.sortedBy { it.name } ?: emptyList()

    fun hasLora(): Boolean = listLoras().isNotEmpty()

    /** 当前生效的 LoRA（优先用户选中的，其次上次选中的，最后取第一个） */
    fun activeLora(): File? {
        val all = listLoras()
        if (all.isEmpty()) return null
        val sel = loraFile
        if (sel != null && sel.isFile && all.any { it.absolutePath == sel.absolutePath }) return sel
        val saved = prefs().getString("selLora", null)
        if (saved != null) all.firstOrNull { it.name == saved }?.let { loraFile = it; return it }
        return all.first()
    }

    /** UI 高亮用：第一个文生图槽位中的 LoRA 名 */
    fun currentLoraName(): String? {
        if (t2iLoraSpinners.isNotEmpty()) {
            return (t2iLoraSpinners.first().selectedItem as? String)
                ?.takeUnless { it == LORA_SLOT_NONE || it.isBlank() }
        }
        return activeLora()?.name
    }

    /** 选中某个 LoRA：同时设成两种生成模式的第一个槽位。 */
    fun selectLora(f: File) {
        loraFile = f
        prefs().edit()
            .putString("selLora", f.name)
            .putString(loraSlotKey("t2i", 0), f.name)
            .putString(loraSlotKey("i2i", 0), f.name)
            .apply()
        refreshLoraHint()
    }

    /** 供「模型」页展示的 LoRA 状态 */
    fun loraSummary(): String {
        val a = activeLora() ?: return c.getString(R.string.s_117)
        val size = "%.1f".format(a.length() / 1048576.0)
        return if (useLora) c.getString(R.string.v_026, (a.nameWithoutExtension), (size))
        else c.getString(R.string.v_027, (a.nameWithoutExtension), (size))
    }

    fun deleteLora(f: File): Boolean = runCatching {
        if (loraFile?.absolutePath == f.absolutePath) loraFile = null
        val ok = f.delete()
        if (ok) {
            val e = prefs().edit()
            for (page in listOf("t2i", "i2i")) {
                for (slot in 0 until MAX_LORA_SLOTS) {
                    val key = loraSlotKey(page, slot)
                    if (prefs().getString(key, null) == f.name) e.putString(key, "")
                }
            }
            if (prefs().getString("selLora", null) == f.name) e.remove("selLora")
            e.apply()
        }
        refreshLoraHint()
        ok
    }.getOrDefault(false)

    private fun refreshLoraHint() {
        try {
            refreshLoraSelectors()
            updateLoraHintText()
            updateI2iLoraHintText()
        } catch (_: Throwable) {}
        syncDrawCore()
    }

    /**
     * 下载 LCM-LoRA。useMirror=true 走 hf-mirror.com（国内可用）。
     * 返回 null 表示成功；否则为错误文案。onProgress(已下载字节, 总字节[未知为 -1])
     */
    suspend fun downloadLora(useMirror: Boolean, onProgress: (Long, Long) -> Unit): String? {
        val host = if (useMirror) "https://hf-mirror.com" else "https://huggingface.co"
        val dest = File(loraDir(), "$LORA_NAME.safetensors")
        val err = downloadToFile("$host/$LORA_HF_PATH", dest, 100L * 1024, onProgress)
        if (err == null) {
            loraFile = dest
            selectNewLoraIfFirstSlotEmpty(dest)
            refreshLoraHint()
        }
        return err
    }

    /** 通用下载：边下边写 .part，完成后改名为目标文件。返回 null = 成功。 */
    private suspend fun downloadToFile(
        url: String,
        dest: File,
        minBytes: Long,
        onProgress: (Long, Long) -> Unit,
    ): String? = withContext(Dispatchers.IO) {
        val tmp = File(dest.parentFile, dest.name + ".part")
        try {
            val conn = (java.net.URL(url).openConnection() as java.net.HttpURLConnection).apply {
                connectTimeout = 20_000
                readTimeout = 60_000
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "Ponko/1.0 (Android)")
            }
            conn.connect()
            val code = conn.responseCode
            if (code !in 200..299) return@withContext c.getString(R.string.v_030, (code))
            val total = conn.contentLengthLong
            conn.inputStream.use { ins ->
                tmp.outputStream().use { outs ->
                    val buf = ByteArray(1 shl 16)
                    var done = 0L
                    var lastTick = 0L
                    while (true) {
                        val r = ins.read(buf)
                        if (r < 0) break
                        outs.write(buf, 0, r)
                        done += r
                        if (done - lastTick > (1 shl 20)) {
                            lastTick = done
                            onProgress(done, total)
                        }
                    }
                    outs.flush()
                }
            }
            if (tmp.length() < minBytes) {
                runCatching { tmp.delete() }
                return@withContext c.getString(R.string.v_031, (tmp.length()))
            }
            if (dest.exists()) dest.delete()
            if (!tmp.renameTo(dest)) {
                tmp.copyTo(dest, overwrite = true)
                runCatching { tmp.delete() }
            }
            onProgress(dest.length(), dest.length())
            null
        } catch (e: Throwable) {
            runCatching { tmp.delete() }
            c.getString(R.string.v_032, (e.message ?: e.javaClass.simpleName))
        }
    }

    // ================= 模型集（单文件 / 多文件；由「模型」页驱动） =================

    /** 一个模型集：单文件 = 只填 model 槽位；多文件 = 主模型 + 各槽位配套文件。 */
    class ModelSet(
        val id: String,
        var name: String,
        var kind: String,
        var family: String,
        val dir: File?,
        val files: MutableMap<String, File> = LinkedHashMap(),
    ) {
        fun mainFile(): File? = files["model"] ?: files["diffusion"] ?: files.values.firstOrNull()
        fun sizeBytes(): Long = files.values.sumOf { it.length() }
        fun slotCount(): Int = files.size
    }

    private fun setsDir(): File = File(File(c.filesDir, DIR_NAME), "sets").apply { mkdirs() }

    private fun safeId(s: String): String {
        val t = s.replace(Regex("[^A-Za-z0-9._-]"), "_").trim('_').take(48)
        return t.ifEmpty { "set" }
    }

    private fun newSetId(base: String): String {
        val b = safeId(base)
        var id = b
        var i = 2
        while (File(setsDir(), id).exists()) { id = "$b-$i"; i++ }
        return id
    }

    private fun writeSetJson(s: ModelSet) {
        val d = s.dir ?: return
        runCatching {
            val o = org.json.JSONObject()
            o.put("name", s.name)
            o.put("kind", s.kind)
            o.put("family", s.family)
            val a = org.json.JSONObject()
            s.files.forEach { (k, f) ->
                val insideSet = f.parentFile?.absoluteFile == d.absoluteFile
                a.put(k, if (insideSet) f.name else f.absolutePath)
            }
            o.put("slots", a)
            File(d, "ponko-set.json").writeText(o.toString())
        }
    }

    private fun readSetJson(dir: File): ModelSet? {
        val o = runCatching { org.json.JSONObject(File(dir, "ponko-set.json").readText()) }.getOrNull() ?: return null
        val files = LinkedHashMap<String, File>()
        o.optJSONObject("slots")?.let { a ->
            val keys = a.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                val fn = a.optString(k, "")
                val stored = File(fn)
                val f = if (stored.isAbsolute) stored else File(dir, fn)
                if (fn.isNotEmpty() && f.isFile) files[k] = f
            }
        }
        return ModelSet(dir.name, o.optString("name", dir.name),
            o.optString("kind", KIND_MULTI), o.optString("family", "custom"), dir, files)
    }

    /** 私有目录里已有的绘图模型：新式模型集（draw/sets 下的子目录）+ 旧版散落的单文件 gguf */
    fun listSets(): List<ModelSet> {
        val out = ArrayList<ModelSet>()
        setsDir().listFiles { f -> f.isDirectory }?.sortedByDescending { it.lastModified() }?.forEach { d ->
            readSetJson(d)?.let { out.add(it) }
        }
        val loose = File(c.filesDir, DIR_NAME).listFiles { f ->
            f.isFile && f.name.endsWith(".gguf", true) && !f.name.startsWith(".")
        }?.sortedByDescending { it.lastModified() } ?: emptyList()
        val looseVae = loose.firstOrNull { it.name.contains("vae", true) }
        for (f in loose) {
            if (f === looseVae) continue
            val files = LinkedHashMap<String, File>()
            files["model"] = f
            if (looseVae != null) files["vae"] = looseVae
            out.add(ModelSet("legacy:${f.name}", f.nameWithoutExtension, KIND_SINGLE, "other", null, files))
        }
        return out
    }

    /** 启动后兜底：把上次选中的模型集（或第一个）设为当前 */
    fun ensureActiveSet() {
        if (activeSet != null) return
        val all = listSets()
        if (all.isEmpty()) return
        val saved = prefs().getString(KEY_SEL_SET, null)
        val pick = all.firstOrNull { it.id == saved } ?: all.first()
        selectSet(pick)
    }

    /** 当前操作的模型集：优先内存里的，其次按偏好里记的 */
    fun activeSetOrSelected(): ModelSet? {
        ensureActiveSet()
        return activeSet
    }

    fun activeSetId(): String? = activeSet?.id

    fun selectSet(s: ModelSet) {
        activeSet = s
        mainModel = s.mainFile()
        vaeModel = s.files["vae"]
        prefs().edit().putString(KEY_SEL_SET, s.id).apply()
    }

    fun deleteSet(s: ModelSet): Boolean {
        val ok = if (s.dir != null) runCatching { s.dir.deleteRecursively() }.getOrDefault(false)
        else s.files.values.all { runCatching { it.delete() }.getOrDefault(false) }
        if (activeSet?.id == s.id) activeSet = null
        if (prefs().getString(KEY_SEL_SET, null) == s.id) prefs().edit().remove(KEY_SEL_SET).apply()
        return ok
    }

    /** 只把槽位从集里摘掉，不删文件（同一个文件可能被多个槽位共用） */
    fun clearSlot(key: String) {
        val s = activeSetOrSelected() ?: return
        s.files.remove(key)
        if (key == "taesd") setTaesdEnabled(false)
        writeSetJson(s)
    }

    /** 模型集摘要：名字 · 单/多文件 · 家族 · 文件数 · 体积 */
    fun setSummary(s: ModelSet): String {
        val kindLabel = c.getString(if (s.kind == KIND_SINGLE) R.string.s_345 else R.string.s_346)
        val fam = c.getString(familyLabelRes(s.family))
        val size = "%.2f GB".format(s.sizeBytes() / 1073741824.0)
        return c.getString(R.string.s_365, s.name, "$kindLabel · $fam", s.slotCount(), size)
    }

    /** 当前模型集里还没填的必填槽位（key） */
    fun slotMissingIds(s: ModelSet): List<String> =
        familySlots(s.kind, s.family).filter { it.second && s.files[it.first] == null }.map { it.first }

    /** 导入一个文件到模型集。
     *  pendingNewSet = true（用户在类型菜单里选了类型）→ 用这个主文件开一个新集；
     *  否则（补配套文件）→ 放进当前集。
     *  返回 null 表示成功，否则为错误文案。 */
    suspend fun importSetFile(
        uri: Uri,
        slotKey: String?,
        onStage: (String) -> Unit,
        onProgress: ((Long, Long) -> Unit)? = null
    ): String? = withContext(Dispatchers.IO) {
        try {
            val name = queryDisplayName(uri) ?: return@withContext c.getString(R.string.s_106)
            val ext = name.substringAfterLast('.', "").lowercase()
            val requestedKey = slotKey ?: if (pendingKind == KIND_SINGLE) "model" else "diffusion"
            val accepted = when (requestedKey) {
                "clip_l" -> ext == "json"
                "clip_g" -> ext == "txt"
                else -> ext == "safetensors"
            }
            if (!accepted) return@withContext c.getString(R.string.draw_hearth_file_type)
            val key = slotKey ?: if (pendingKind == KIND_SINGLE) "model" else "diffusion"
            val cur = activeSet
            val useSet: ModelSet = when {
                pendingNewSet || cur == null -> {
                    val id = newSetId(name.substringBeforeLast('.'))
                    val dir = File(setsDir(), id).apply { mkdirs() }
                    ModelSet(id, id, pendingKind, pendingFamily, dir).also {
                        activeSet = it
                        pendingNewSet = false
                    }
                }
                cur.dir == null -> {
                    // Upgrade a legacy loose-file model to a model set without copying multi-GB
                    // weights. Persist external paths so adding TAESD never replaces the main model.
                    val id = newSetId(cur.name)
                    val dir = File(setsDir(), id).apply { mkdirs() }
                    ModelSet(id, cur.name, cur.kind, cur.family, dir, LinkedHashMap(cur.files)).also {
                        activeSet = it
                    }
                }
                else -> cur
            }
            val dir: File = useSet.dir ?: return@withContext c.getString(R.string.s_106)

            withContext(Dispatchers.Main) { onStage(c.getString(R.string.v_033, name)) }
            val dest = File(dir, name)
            val total = runCatching {
                c.contentResolver.openAssetFileDescriptor(uri, "r")?.use { it.length } ?: -1L
            }.getOrDefault(-1L)
            withContext(Dispatchers.Main) { onProgress?.invoke(0L, total) }
            c.contentResolver.openInputStream(uri)?.use { ins ->
                dest.outputStream().use { outs ->
                    val buf = ByteArray(1 shl 20)
                    var done = 0L
                    var tick = 0L
                    while (true) {
                        val n = ins.read(buf)
                        if (n < 0) break
                        outs.write(buf, 0, n)
                        done += n
                        if (done - tick >= (1L shl 20)) {
                            tick = done
                            val dd = done
                            withContext(Dispatchers.Main) { onProgress?.invoke(dd, total) }
                        }
                    }
                    val dd = done
                    withContext(Dispatchers.Main) { onProgress?.invoke(dd, total) }
                }
            } ?: return@withContext c.getString(R.string.s_106)

            useSet.files[key] = dest
            writeSetJson(useSet)
            selectSet(useSet)
            if (key == "taesd") setTaesdEnabled(true)
            withContext(Dispatchers.Main) {
                statusText.text = c.getString(R.string.s_161)
            }
            null
        } catch (e: Throwable) {
            c.getString(R.string.v_038, (e.message ?: e.javaClass.simpleName))
        }
    }

    /** 从单文件导入本地 LoRA（.safetensors）。返回 null 表示成功。 */
    suspend fun importLoraFile(
        uri: Uri,
        onStage: (String) -> Unit,
        onProgress: ((Long, Long) -> Unit)? = null
    ): String? = withContext(Dispatchers.IO) {
        try {
            val name = queryDisplayName(uri)
            if (name == null || !name.endsWith(".safetensors", ignoreCase = true)) {
                return@withContext c.getString(R.string.s_218)
            }
            withContext(Dispatchers.Main) { onStage(c.getString(R.string.v_034, name)) }
            val dest = File(loraDir(), name)
            val total = runCatching {
                c.contentResolver.openAssetFileDescriptor(uri, "r")?.use { it.length } ?: -1L
            }.getOrDefault(-1L)
            withContext(Dispatchers.Main) { onProgress?.invoke(0L, total) }
            c.contentResolver.openInputStream(uri)?.use { ins ->
                dest.outputStream().use { outs ->
                    val buf = ByteArray(1 shl 20)
                    var done = 0L
                    var tick = 0L
                    while (true) {
                        val n = ins.read(buf)
                        if (n < 0) break
                        outs.write(buf, 0, n)
                        done += n
                        if (done - tick >= (1L shl 20)) {
                            tick = done
                            val d = done
                            withContext(Dispatchers.Main) { onProgress?.invoke(d, total) }
                        }
                    }
                }
            } ?: return@withContext c.getString(R.string.s_106)
            withContext(Dispatchers.Main) {
                loraFile = dest
                selectNewLoraIfFirstSlotEmpty(dest)
                refreshLoraHint()
            }
            null
        } catch (e: Throwable) {
            c.getString(R.string.v_038, (e.message ?: e.javaClass.simpleName))
        }
    }

    /** 从 content:// URI 取显示文件名 */
    private fun queryDisplayName(uri: Uri): String? =
        c.contentResolver.query(uri, null, null, null, null)?.use { cur ->
            val idx = cur.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (idx >= 0 && cur.moveToFirst()) cur.getString(idx) else null
        }

    /** 加载当前选中的模型集（「加载绘图模型」按钮） */
    suspend fun loadSelectedSet(): String? = withContext(Dispatchers.IO) {
        try {
            loadSet(activeSetOrSelected())
        } catch (e: Throwable) {
            runCatching {
                File(File(c.filesDir, DIR_NAME), ".loadstage")
                    .appendText(c.getString(R.string.v_039, (e.javaClass.name), (e.message)) +
                        android.util.Log.getStackTraceString(e) + "\n")
            }
            c.getString(R.string.v_040, (e.message ?: e.javaClass.simpleName))
        }
    }

    /** 当前模型集的主模型文件名 */
    fun currentMainName(): String? = activeSetOrSelected()?.name

    /** 某个文件的量化等级（Q4_0 / Q8_0 / …），读不出来返回 null */
    fun quantOf(f: File): String? = if (f.extension.equals("safetensors", ignoreCase = true)) "SafeTensors" else null

    /** 删除单个已复制的模型文件（旧接口，保留给兼容用） */
    fun deleteModel(f: File): Boolean = runCatching { f.delete() }.getOrDefault(false)

    /** 私有目录里散落的 gguf（旧版遗留；新式模型请用 listSets） */
    fun listModels(): List<File> =
        File(c.filesDir, DIR_NAME)
            .listFiles { f -> f.isFile && f.name.endsWith(".safetensors", ignoreCase = true) }
            ?.sortedByDescending { it.lastModified() } ?: emptyList()

    private suspend fun loadSet(set: ModelSet?): String? {
        if (set == null) {
            statusText.post { statusText.text = c.getString(R.string.s_114) }
            return c.getString(R.string.s_114)
        }
        if (DrawCore.busy || DrawCore.uiBusy) {
            val message = c.getString(R.string.s_428)
            statusText.post { statusText.text = message }
            return message
        }

        val slots = LinkedHashMap<String, String>()
        set.files.forEach { (key, file) -> slots[key] = file.absolutePath }
        val missing = slotMissingIds(set).map { c.getString(slotLabelRes(it)) }
        if (missing.isNotEmpty()) {
            val message = c.getString(R.string.s_363, missing.joinToString("、"))
            statusText.post { statusText.text = message }
            return message
        }

        val root = File(c.filesDir, DIR_NAME)
        val stageFile = File(root, ".loadstage")
        fun stage(message: String) { runCatching { stageFile.appendText("$message\n") } }
        runCatching { stageFile.writeText("") }
        stage("Hearth / Burn-WGPU Vulkan")
        stage("Family: ${set.family}; model set: ${set.name}")
        slots.forEach { (key, path) -> stage("$key = ${File(path).name} (${File(path).length() / 1048576} MB)") }
        stage("Device: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}; ABI: ${android.os.Build.SUPPORTED_ABIS.firstOrNull()}")
        stage("Heap: ${Runtime.getRuntime().maxMemory() / 1048576} MB; storage: ${root.usableSpace / 1048576} MB")

        return try {
            // Detach the process-wide API reference before freeing its native handle. If loading
            // the replacement fails, the API must stay detached instead of calling a stale handle.
            DrawCore.release()
            if (sdHandle != 0L) runCatching { HearthEngine.nativeFree(sdHandle) }
            sdHandle = 0L
            stage(HearthEngine.info())
            val handle = HearthEngine.create(
                modelPath = slots["model"].orEmpty(),
                vaePath = slots["vae"],
                nThreads = nThreads,
                wtype = HearthEngine.WTYPE_KEEP,
                flashAttn = false,
                vaeConvDirect = false,
                backend = "Vulkan",
                slots = HearthEngine.buildSlots(slots),
                family = set.family,
            )
            if (handle == 0L) {
                stage(HearthEngine.lastParams())
                runCatching { stageFile.delete() }
                return c.getString(R.string.s_059)
            }

            sdHandle = handle
            activeBackend = "Vulkan"
            activeVulkanDiagnostic = HearthEngine.vulkanDiagnostics()
            mainModel = set.mainFile()
            vaeModel = set.files["vae"]
            DrawCore.attach(handle, mainModel?.name ?: set.name, "Vulkan")
            stage("Hearth context ready")
            stage(activeVulkanDiagnostic)
            val summary = buildString {
                append(c.getString(R.string.s_094)).append(set.name)
                append(" · Hearth · Burn/WGPU Vulkan")
            }
            statusText.post { statusText.text = summary }
            statusText.post { refreshQuantText() }
            statusText.post { runCatching { syncDrawCore() } }
            runCatching { stageFile.delete() }
            onPipelineReady?.invoke()
            null
        } catch (e: Throwable) {
            stage("${e.javaClass.name}: ${e.message}")
            runCatching { stageFile.appendText(android.util.Log.getStackTraceString(e) + "\n") }
            c.getString(R.string.v_047, (e.message ?: e.javaClass.simpleName))
        }
    }

    fun unloadModel() {
        if (DrawCore.busy || DrawCore.uiBusy) {
            throw IllegalStateException(c.getString(R.string.s_428))
        }
        DrawCore.release()   // 先清引用，别让 API 拿到已经 free 的句柄
        if (sdHandle != 0L) runCatching { HearthEngine.nativeFree(sdHandle) }
        sdHandle = 0L
        activeBackend = "CPU"
        activeVulkanDiagnostic = ""
        try {
            statusText.text = c.getString(R.string.s_115)
        } catch (_: Throwable) {}
        refreshQuantText()
    }

    fun hasModel(): Boolean = listSets().isNotEmpty()

    fun modelSummary(): String = when {
        sdHandle != 0L -> {
            val q = mainModel?.let { quantOf(it) }
            c.getString(R.string.s_094) + (mainModel?.name ?: c.getString(R.string.s_155)) +
                (if (q != null) "（$q）" else "") +
                " · Hearth / Burn-WGPU Vulkan"
        }
        hasModel() -> (activeSetOrSelected()?.name?.let { it + " · " } ?: "") + c.getString(R.string.s_088)
        else -> c.getString(R.string.s_115)
    }

    fun refreshStatus() {
        refreshModelCard()
    }

    /** 顶部模型卡片跟随子页切换：显示当前绘图模型 */
    private fun refreshModelCard() {
        if (!::modelCardTitle.isInitialized) return
        if (::quantizeUnetButton.isInitialized) quantizeUnetButton.visibility = View.GONE
        if (::resAdapterButton.isInitialized) resAdapterButton.visibility = View.GONE
        if (::fuseLoraButton.isInitialized) fuseLoraButton.visibility = View.GONE
        modelCardTitle.text = c.getString(R.string.s_164)
        try { statusText.text = modelSummary() } catch (_: Throwable) {}
        refreshQuantText()
    }

        /** 当前绘图模型的量化等级（始终可见）。
     *  量化等级是烧在模型文件里的（Q4_0 的 gguf 就是 Q4_0），App 无法凭空转换，只能读出来展示。 */
    private fun refreshQuantText() {
        try {
            val m = mainModel ?: activeSetOrSelected()?.mainFile() ?: listModels().maxByOrNull { it.length() }
            quantText.text = if (m == null) {
                c.getString(R.string.s_103)
            } else {
                val q = quantOf(m) ?: c.getString(R.string.s_119)
                c.getString(R.string.v_048, (q), (m.name))
            }
        } catch (_: Throwable) {}
    }

    // ================= UNet-only quantizer =================

    private fun showUnetQuantizer() {
        if (generating || DrawCore.busy || DrawCore.uiBusy) {
            Toast.makeText(c, c.getString(R.string.draw_unet_quantize_busy), Toast.LENGTH_LONG).show()
            return
        }
        val set = activeSetOrSelected()
        if (set == null) {
            Toast.makeText(c, c.getString(R.string.draw_unet_quantize_no_model), Toast.LENGTH_LONG).show()
            return
        }
        if (set.family !in setOf(FAM_SD15, FAM_SDXL)) {
            Toast.makeText(c, c.getString(R.string.draw_unet_quantize_not_supported), Toast.LENGTH_LONG).show()
            return
        }
        // Single-file checkpoints use "model"; split Diffusers-style sets use "diffusion".
        val sourceSlot = if (set.files["diffusion"]?.isFile == true) "diffusion" else "model"
        val source = set.files[sourceSlot]
        if (source == null || !source.isFile) {
            Toast.makeText(c, c.getString(R.string.draw_unet_quantize_no_unet), Toast.LENGTH_LONG).show()
            return
        }
        val extension = source.extension.lowercase()
        if (extension !in setOf("safetensors", "sft", "ckpt", "pt", "pth")) {
            Toast.makeText(c, c.getString(R.string.draw_unet_quantize_source_type), Toast.LENGTH_LONG).show()
            return
        }

        val types = arrayOf("q4_0", "q5_0", "q8_0")
        val labels = arrayOf("Q4_0 · 4-bit", "Q5_0 · 5-bit", "Q8_0 · 8-bit")
        android.app.AlertDialog.Builder(act)
            .setTitle(c.getString(R.string.draw_unet_quantize_title))
            .setItems(labels) { chooser, which ->
                chooser.dismiss()
                val type = types[which]
                android.app.AlertDialog.Builder(act)
                    .setTitle(c.getString(R.string.draw_unet_quantize_title))
                    .setMessage(c.getString(R.string.draw_unet_quantize_confirm, type.uppercase()))
                    .setPositiveButton(c.getString(R.string.draw_unet_quantize_start)) { _, _ ->
                        runUnetQuantization(set, sourceSlot, source, type)
                    }
                    .setNegativeButton(c.getString(R.string.s_066), null)
                    .show()
            }
            .setNegativeButton(c.getString(R.string.s_066), null)
            .show()
    }

    private fun runUnetQuantization(set: ModelSet, sourceSlot: String, source: File, quantType: String) {
        val id = newSetId(set.id + "-" + quantType + "-unet")
        val outputDir = File(setsDir(), id)
        if (!outputDir.mkdirs()) {
            Toast.makeText(c, c.getString(R.string.draw_unet_quantize_create_failed), Toast.LENGTH_LONG).show()
            return
        }
        val outputFile = File(outputDir, source.nameWithoutExtension + "-unet-" + quantType + ".gguf")

        val progressView = LinearLayout(c).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(24), dp(18), dp(24), dp(18))
            addView(ProgressBar(c).apply { isIndeterminate = true },
                LinearLayout.LayoutParams(dp(32), dp(32)))
            addView(TextView(c).apply {
                text = c.getString(R.string.draw_unet_quantize_working, quantType.uppercase())
                textSize = 14f
                setTextColor(pal.text)
                setPadding(dp(16), 0, 0, 0)
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }
        val progressDialog = android.app.AlertDialog.Builder(act)
            .setTitle(c.getString(R.string.draw_unet_quantize_title))
            .setView(progressView)
            .setCancelable(false)
            .create()
        progressDialog.show()

        scope.launch {
            try {
                // Release mapped weights first so the converter has more room on low-RAM phones.
                unloadModel()
                val success = withContext(Dispatchers.IO) {
                    HearthEngine.nativeQuantizeUnet(
                        source.absolutePath,
                        outputFile.absolutePath,
                        quantType,
                        sourceSlot == "diffusion",
                        nThreads.coerceIn(1, 6),
                    )
                }
                if (!success || !outputFile.isFile || outputFile.length() <= 0L) {
                    throw IllegalStateException(c.getString(R.string.draw_unet_quantize_failed_detail))
                }

                val files = LinkedHashMap(set.files)
                files[sourceSlot] = outputFile
                val quantizedSet = ModelSet(
                    id,
                    set.name + " (" + quantType.uppercase() + " UNet)",
                    set.kind,
                    set.family,
                    outputDir,
                    files,
                )
                withContext(Dispatchers.IO) { writeSetJson(quantizedSet) }
                if (!File(outputDir, "ponko-set.json").isFile) {
                    throw IllegalStateException(c.getString(R.string.draw_unet_quantize_register_failed))
                }
                selectSet(quantizedSet)
                refreshModelCard()
                Toast.makeText(c, c.getString(R.string.draw_unet_quantize_success, quantType.uppercase()), Toast.LENGTH_LONG).show()
            } catch (e: Throwable) {
                withContext(Dispatchers.IO) { runCatching { outputDir.deleteRecursively() } }
                Toast.makeText(
                    c,
                    c.getString(R.string.draw_unet_quantize_failed, e.message ?: e.javaClass.simpleName),
                    Toast.LENGTH_LONG,
                ).show()
            } finally {
                progressDialog.dismiss()
            }
        }
    }



    // ================= Native LoRA fusion =================

    private fun showLoraFusionWizard(preselectName: String? = null) {
        if (generating || DrawCore.busy || DrawCore.uiBusy) {
            Toast.makeText(c, c.getString(R.string.draw_lora_fusion_busy), Toast.LENGTH_LONG).show()
            return
        }
        val set = activeSetOrSelected()
        if (set == null) {
            toast(c.getString(R.string.draw_lora_fusion_no_model))
            return
        }
        if (set.kind != KIND_SINGLE || set.family !in setOf(FAM_SD15, FAM_SDXL)) {
            toast(c.getString(R.string.draw_lora_fusion_supported))
            return
        }
        val source = set.files["model"] ?: set.mainFile()
        if (source == null || !source.isFile ||
            source.extension.lowercase() !in setOf("safetensors", "gguf", "ckpt", "pt", "pth")) {
            toast(c.getString(R.string.draw_lora_fusion_supported))
            return
        }

        val available = listLoras()
        val checkboxes = mutableListOf<CheckBox>()
        val content = LinearLayout(c).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(8), dp(20), dp(8))
        }
        val intro = if (available.isEmpty()) {
            c.getString(R.string.draw_lora_fusion_no_loras)
        } else {
            c.getString(R.string.draw_lora_fusion_intro, source.name)
        }
        content.addView(body(intro), matchWrap(bottom = if (available.isEmpty()) 0 else 8))

        for (lora in available) {
            val checkbox = CheckBox(c).apply {
                text = lora.name
                textSize = 13f
                setTextColor(pal.text)
                maxLines = 2
                isChecked = lora.name == preselectName
                setPadding(0, dp(4), 0, dp(4))
            }
            checkboxes.add(checkbox)
            content.addView(checkbox, matchWrap(bottom = 2))
        }

        val scroll = ScrollView(c).apply {
            isFillViewport = true
            addView(content, ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ))
        }
        val dialog = android.app.AlertDialog.Builder(act)
            .setTitle(c.getString(R.string.draw_lora_fusion_title))
            .setView(scroll)
            .setPositiveButton(c.getString(R.string.draw_lora_fusion_next), null)
            .setNeutralButton(c.getString(R.string.draw_lora_fusion_import)) { _, _ ->
                pickLoraForFusion()
            }
            .setNegativeButton(c.getString(R.string.s_066), null)
            .create()
        dialog.setOnShowListener {
            val next = dialog.getButton(android.app.AlertDialog.BUTTON_POSITIVE)
            next.isEnabled = available.isNotEmpty()
            next.setOnClickListener {
                val selected = available.filterIndexed { index, _ -> checkboxes[index].isChecked }
                if (selected.isEmpty()) {
                    toast(c.getString(R.string.draw_lora_fusion_no_loras))
                } else {
                    dialog.dismiss()
                    showLoraFusionSettings(set, source, selected)
                }
            }
        }
        dialog.show()
    }

    private fun pickLoraForFusion() {
        val intent = android.content.Intent(android.content.Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(android.content.Intent.CATEGORY_OPENABLE)
            type = "*/*"
            putExtra(android.content.Intent.EXTRA_TITLE, c.getString(R.string.draw_lora_fusion_pick_file))
        }
        act.startActivityForResult(intent, REQ_LORA_FUSION_IMPORT)
    }

    private fun handleLoraFusionPickedFile(
        requestCode: Int,
        resultCode: Int,
        data: android.content.Intent?,
    ): Boolean {
        if (requestCode != REQ_LORA_FUSION_IMPORT) return false
        if (resultCode != Activity.RESULT_OK) return true
        val uri = data?.data
        if (uri == null) {
            toast(c.getString(R.string.draw_lora_fusion_import_failed, c.getString(R.string.s_106)))
            return true
        }
        val name = queryDisplayName(uri)?.takeIf { it.isNotBlank() } ?: "lora.safetensors"
        if (!name.endsWith(".safetensors", ignoreCase = true)) {
            toast(c.getString(R.string.draw_lora_fusion_bad_lora))
            return true
        }

        scope.launch {
            try {
                val imported = withContext(Dispatchers.IO) {
                    val safeName = File(name).name
                        .replace(Regex("[^A-Za-z0-9._ -]"), "_")
                        .take(120)
                        .ifBlank { "lora.safetensors" }
                    val stem = safeName.substringBeforeLast('.', safeName)
                    val ext = safeName.substringAfterLast('.', "")
                    val dir = loraDir()
                    var destination = File(dir, safeName)
                    var suffix = 2
                    while (destination.exists()) {
                        destination = File(dir, "$stem-$suffix" + if (ext.isEmpty()) "" else ".$ext")
                        suffix++
                    }
                    try {
                        copyContentUri(uri, destination)
                        destination
                    } catch (e: Throwable) {
                        destination.delete()
                        throw e
                    }
                }
                toast(c.getString(R.string.draw_lora_fusion_imported, imported.name))
                // Reopen with the imported adapter selected, so Next can use it immediately.
                showLoraFusionWizard(preselectName = imported.name)
            } catch (e: Throwable) {
                toast(c.getString(
                    R.string.draw_lora_fusion_import_failed,
                    e.message ?: e.javaClass.simpleName,
                ))
            }
        }
        return true
    }

    private fun showLoraFusionSettings(baseSet: ModelSet, source: File, loras: List<File>) {
        val panel = LinearLayout(c).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(8), dp(20), dp(8))
        }
        panel.addView(body(c.getString(R.string.draw_lora_fusion_strength_hint)), matchWrap(bottom = 6))
        val scaleFields = mutableListOf<EditText>()
        for (lora in loras) {
            val row = LinearLayout(c).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }
            row.addView(TextView(c).apply {
                text = lora.name
                textSize = 11.5f
                setTextColor(textColor)
                maxLines = 2
            }, LinearLayout.LayoutParams(0, dp(48), 1f))
            val scale = smallNumber("1.0")
            scale.setText("1.0")
            scale.hint = "1.0"
            scale.contentDescription = c.getString(R.string.draw_lora_fusion_strength)
            row.addView(scale, LinearLayout.LayoutParams(dp(78), dp(46)).apply {
                leftMargin = dp(8)
            })
            scaleFields.add(scale)
            panel.addView(row)
        }

        panel.addView(TextView(c).apply {
            text = c.getString(R.string.draw_lora_fusion_output_precision)
            textSize = 12.5f
            setTextColor(textColor)
            setPadding(0, dp(12), 0, dp(4))
        })
        val quantTypes = listOf("f16", "q4_0", "q5_0", "q8_0")
        val quantLabels = listOf(
            "F16 · no new quantization",
            "Q4_0 · quantize UNet",
            "Q5_0 · quantize UNet",
            "Q8_0 · quantize UNet",
        )
        val quantSpinner = android.widget.Spinner(c).apply {
            adapter = android.widget.ArrayAdapter(
                c, android.R.layout.simple_spinner_item, quantLabels
            ).apply { setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
            setSelection(0, false)
        }
        panel.addView(quantSpinner, matchWrap())

        val scroll = ScrollView(c).apply { addView(panel) }
        val dialog = android.app.AlertDialog.Builder(act)
            .setTitle(c.getString(R.string.draw_lora_fusion_settings))
            .setView(scroll)
            .setPositiveButton(c.getString(R.string.draw_lora_fusion_apply), null)
            .setNegativeButton(c.getString(R.string.s_066), null)
            .create()
        dialog.setOnShowListener {
            dialog.getButton(android.app.AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val scales = scaleFields.map { field ->
                    field.text.toString().trim().toFloatOrNull()
                }
                if (scales.any { it == null || !it.isFinite() || it < -4f || it > 4f }) {
                    toast(c.getString(R.string.draw_lora_fusion_bad_scale))
                    return@setOnClickListener
                }
                val selectedScales = scales.map { it!! }
                val quantType = quantTypes.getOrElse(quantSpinner.selectedItemPosition) { "f16" }
                dialog.dismiss()
                runLoraFusion(baseSet, source, loras, selectedScales, quantType)
            }
        }
        dialog.show()
    }

    private fun runLoraFusion(
        baseSet: ModelSet,
        source: File,
        loras: List<File>,
        scales: List<Float>,
        quantType: String,
    ) {
        if (generating || DrawCore.busy || DrawCore.uiBusy) {
            toast(c.getString(R.string.draw_lora_fusion_busy))
            return
        }
        val setId = newSetId(baseSet.name + "-lora-fused")
        val outputDir = File(setsDir(), setId)
        if (!outputDir.mkdirs()) {
            toast(c.getString(R.string.draw_lora_fusion_create_failed))
            return
        }
        val outputFile = File(outputDir, source.nameWithoutExtension + "-fused-" + quantType + ".gguf")
        val progressView = LinearLayout(c).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(24), dp(18), dp(24), dp(18))
            addView(ProgressBar(c).apply { isIndeterminate = true },
                LinearLayout.LayoutParams(dp(32), dp(32)))
            addView(TextView(c).apply {
                text = c.getString(R.string.draw_lora_fusion_working, quantType.uppercase())
                textSize = 14f
                setTextColor(pal.text)
                setPadding(dp(16), 0, 0, 0)
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }
        val dialog = android.app.AlertDialog.Builder(act)
            .setTitle(c.getString(R.string.draw_lora_fusion_title))
            .setView(progressView)
            .setCancelable(false)
            .create()
        dialog.show()

        scope.launch {
            try {
                // Release mapped checkpoint weights first. The native converter streams the
                // source tensor-by-tensor and publishes a new file without editing its inputs.
                unloadModel()
                val success = withContext(Dispatchers.IO) {
                    HearthEngine.nativeFuseLoras(
                        inputPath = source.absolutePath,
                        loraPaths = loras.map { it.absolutePath },
                        loraScales = scales,
                        outputPath = outputFile.absolutePath,
                        quantType = quantType,
                        nThreads = nThreads.coerceIn(1, 6),
                    )
                }
                if (!success || !outputFile.isFile || outputFile.length() <= 0L) {
                    throw IllegalStateException(c.getString(R.string.draw_lora_fusion_failed_detail))
                }

                val files = LinkedHashMap(baseSet.files)
                files["model"] = outputFile
                val fusedSet = ModelSet(
                    setId,
                    baseSet.name + " (LoRA fused)",
                    KIND_SINGLE,
                    baseSet.family,
                    outputDir,
                    files,
                )
                withContext(Dispatchers.IO) { writeSetJson(fusedSet) }
                if (!File(outputDir, "ponko-set.json").isFile) {
                    throw IllegalStateException(c.getString(R.string.draw_lora_fusion_register_failed))
                }
                selectSet(fusedSet)
                refreshModelCard()
                Toast.makeText(c, c.getString(R.string.draw_lora_fusion_success, quantType.uppercase()),
                    Toast.LENGTH_LONG).show()
            } catch (e: Throwable) {
                withContext(Dispatchers.IO) { runCatching { outputDir.deleteRecursively() } }
                Toast.makeText(
                    c,
                    c.getString(R.string.draw_lora_fusion_failed, e.message ?: e.javaClass.simpleName),
                    Toast.LENGTH_LONG,
                ).show()
            } finally {
                dialog.dismiss()
            }
        }
    }

    // ================= ResAdapter (LoRA + normalization weights) =================

    private fun showResAdapterWizard() {
        if (generating || DrawCore.busy || DrawCore.uiBusy) {
            Toast.makeText(c, c.getString(R.string.draw_resadapter_busy), Toast.LENGTH_LONG).show()
            return
        }
        val set = activeSetOrSelected()
        if (set == null) {
            toast(c.getString(R.string.draw_resadapter_no_model))
            return
        }
        if (set.family != FAM_SDXL || set.kind != KIND_SINGLE) {
            toast(c.getString(R.string.draw_resadapter_supported_models))
            return
        }
        val source = set.files["model"] ?: set.mainFile()
        if (source == null || !source.isFile ||
            source.extension.lowercase() !in setOf("safetensors", "gguf", "ckpt", "pt")) {
            toast(c.getString(R.string.draw_resadapter_supported_models))
            return
        }

        val types = arrayOf("q4_0", "q5_0", "q8_0")
        val labels = arrayOf("Q4_0 · 4-bit UNet", "Q5_0 · 5-bit UNet", "Q8_0 · 8-bit UNet")
        android.app.AlertDialog.Builder(act)
            .setTitle(c.getString(R.string.draw_resadapter_title))
            .setMessage(c.getString(R.string.draw_resadapter_intro, source.name))
            .setItems(labels) { dialog, which ->
                dialog.dismiss()
                pendingResAdapterSet = set
                pendingResAdapterQuantType = types[which]
                pendingResAdapterLoraUri = null
                pickResAdapterFile(REQ_RESADAPTER_LORA, R.string.draw_resadapter_pick_lora)
            }
            .setNegativeButton(c.getString(R.string.s_066), null)
            .show()
    }

    private fun pickResAdapterFile(requestCode: Int, titleRes: Int) {
        val intent = android.content.Intent(android.content.Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(android.content.Intent.CATEGORY_OPENABLE)
            type = "*/*"
            putExtra(android.content.Intent.EXTRA_TITLE, c.getString(titleRes))
        }
        act.startActivityForResult(intent, requestCode)
    }

    private fun resetPendingResAdapter() {
        pendingResAdapterSet = null
        pendingResAdapterLoraUri = null
        pendingResAdapterQuantType = "q5_0"
    }

    private fun copyContentUri(uri: Uri, destination: File) {
        destination.parentFile?.mkdirs()
        val input = c.contentResolver.openInputStream(uri)
            ?: throw IllegalStateException(c.getString(R.string.draw_resadapter_copy_failed))
        input.use { source ->
            destination.outputStream().buffered().use { output ->
                source.copyTo(output, 1 shl 20)
            }
        }
        if (!destination.isFile || destination.length() <= 0L) {
            throw IllegalStateException(c.getString(R.string.draw_resadapter_copy_failed))
        }
    }

    private fun handleResAdapterPickedFile(requestCode: Int, resultCode: Int, data: android.content.Intent?): Boolean {
        if (requestCode != REQ_RESADAPTER_LORA && requestCode != REQ_RESADAPTER_NORM) return false
        if (resultCode != Activity.RESULT_OK) {
            resetPendingResAdapter()
            return true
        }
        val uri = data?.data
        if (uri == null) {
            resetPendingResAdapter()
            toast(c.getString(R.string.draw_resadapter_copy_failed))
            return true
        }
        val fileName = queryDisplayName(uri).orEmpty()
        if (!fileName.endsWith(".safetensors", ignoreCase = true)) {
            resetPendingResAdapter()
            toast(c.getString(if (requestCode == REQ_RESADAPTER_LORA)
                R.string.draw_resadapter_bad_lora else R.string.draw_resadapter_bad_norm))
            return true
        }

        if (requestCode == REQ_RESADAPTER_LORA) {
            pendingResAdapterLoraUri = uri
            pickResAdapterFile(REQ_RESADAPTER_NORM, R.string.draw_resadapter_pick_norm)
            return true
        }

        val set = pendingResAdapterSet
        val loraUri = pendingResAdapterLoraUri
        val quantType = pendingResAdapterQuantType
        if (set == null || loraUri == null) {
            resetPendingResAdapter()
            toast(c.getString(R.string.draw_resadapter_no_model))
            return true
        }
        val loraName = queryDisplayName(loraUri) ?: "resadapter_lora.safetensors"
        val normName = fileName
        resetPendingResAdapter()

        android.app.AlertDialog.Builder(act)
            .setTitle(c.getString(R.string.draw_resadapter_confirm_title))
            .setMessage(c.getString(
                R.string.draw_resadapter_confirm,
                set.name,
                loraName,
                normName,
                quantType.uppercase(),
            ))
            .setPositiveButton(c.getString(R.string.draw_resadapter_apply)) { _, _ ->
                runResAdapterApply(set, loraUri, loraName, uri, normName, quantType)
            }
            .setNegativeButton(c.getString(R.string.s_066), null)
            .show()
        return true
    }

    private fun runResAdapterApply(
        baseSet: ModelSet,
        loraUri: Uri,
        loraName: String,
        normUri: Uri,
        normName: String,
        quantType: String,
    ) {
        val source = baseSet.files["model"] ?: baseSet.mainFile()
        if (source == null || !source.isFile) {
            toast(c.getString(R.string.draw_resadapter_no_model))
            return
        }
        val setId = newSetId(baseSet.name + "-resadapter-" + quantType)
        val outputDir = File(setsDir(), setId)
        if (!outputDir.mkdirs()) {
            toast(c.getString(R.string.draw_resadapter_create_failed))
            return
        }

        val progressView = LinearLayout(c).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(24), dp(18), dp(24), dp(18))
            addView(ProgressBar(c).apply { isIndeterminate = true },
                LinearLayout.LayoutParams(dp(32), dp(32)))
            addView(TextView(c).apply {
                text = c.getString(R.string.draw_resadapter_working, quantType.uppercase())
                textSize = 14f
                setTextColor(pal.text)
                setPadding(dp(16), 0, 0, 0)
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }
        val dialog = android.app.AlertDialog.Builder(act)
            .setTitle(c.getString(R.string.draw_resadapter_title))
            .setView(progressView)
            .setCancelable(false)
            .create()
        dialog.show()

        scope.launch {
            val workDir = File(c.cacheDir, "resadapter-" + System.currentTimeMillis())
            var installedLora: File? = null
            try {
                val stagedLora = File(workDir, "resadapter_lora.safetensors")
                val stagedNorm = File(workDir, "resadapter_normalization.safetensors")
                val outputFile = File(outputDir, source.nameWithoutExtension + "-resadapter-" + quantType + ".gguf")

                withContext(Dispatchers.IO) {
                    workDir.mkdirs()
                    copyContentUri(loraUri, stagedLora)
                    copyContentUri(normUri, stagedNorm)
                }

                // Free model mappings before streaming the full SDXL checkpoint through the converter.
                unloadModel()
                val ok = withContext(Dispatchers.IO) {
                    HearthEngine.nativeApplyResAdapter(
                        inputModelPath = source.absolutePath,
                        vaePath = baseSet.files["vae"]?.takeIf { it.isFile }?.absolutePath,
                        normalizationPath = stagedNorm.absolutePath,
                        outputPath = outputFile.absolutePath,
                        quantType = quantType,
                        nThreads = nThreads.coerceIn(1, 6),
                    )
                }
                if (!ok || !outputFile.isFile || outputFile.length() <= 0L) {
                    throw IllegalStateException(c.getString(R.string.draw_resadapter_failed_detail))
                }

                val finalLora = withContext(Dispatchers.IO) {
                    val rawStem = File(loraName).nameWithoutExtension
                        .replace(Regex("[^A-Za-z0-9._-]"), "_")
                        .take(80).ifBlank { "resadapter" }
                    var target = File(loraDir(), "$rawStem-resadapter.safetensors")
                    var suffix = 2
                    while (target.exists()) {
                        target = File(loraDir(), "$rawStem-resadapter-$suffix.safetensors")
                        suffix++
                    }
                    stagedLora.copyTo(target, overwrite = false)
                    target
                }
                installedLora = finalLora

                val nextFiles = LinkedHashMap<String, File>()
                nextFiles["model"] = outputFile
                baseSet.files["taesd"]?.takeIf { it.isFile }?.let { nextFiles["taesd"] = it }
                val resultSet = ModelSet(
                    setId,
                    baseSet.name + " (ResAdapter " + quantType.uppercase() + ")",
                    KIND_SINGLE,
                    FAM_SDXL,
                    outputDir,
                    nextFiles,
                )
                withContext(Dispatchers.IO) { writeSetJson(resultSet) }
                if (!File(outputDir, "ponko-set.json").isFile) {
                    throw IllegalStateException(c.getString(R.string.draw_resadapter_register_failed))
                }

                selectSet(resultSet)
                useLora = true
                prefs().edit()
                    .putString("selLora", finalLora.name)
                    .putString(loraSlotKey("t2i", 0), finalLora.name)
                    .putString(loraSlotKey("i2i", 0), finalLora.name)
                    .putString(loraWeightKey("t2i", 0), "1.0")
                    .putString(loraWeightKey("i2i", 0), "1.0")
                    .apply()
                refreshLoraSelectors()
                refreshModelCard()
                Toast.makeText(c, c.getString(R.string.draw_resadapter_success, quantType.uppercase()), Toast.LENGTH_LONG).show()
            } catch (e: Throwable) {
                withContext(Dispatchers.IO) {
                    runCatching { outputDir.deleteRecursively() }
                    installedLora?.let { runCatching { it.delete() } }
                }
                Toast.makeText(
                    c,
                    c.getString(R.string.draw_resadapter_failed, e.message ?: e.javaClass.simpleName),
                    Toast.LENGTH_LONG,
                ).show()
            } finally {
                withContext(Dispatchers.IO) { runCatching { workDir.deleteRecursively() } }
                dialog.dismiss()
            }
        }
    }

    // ================= 生成 =================

    private fun generateFromUi() {
        persistAll()   // 生成前先把参数落盘，保证「改完就生成」也被记住
        if (!isReady()) {
            val msg = if (llmLoaded) c.getString(R.string.s_100)
            else c.getString(R.string.s_173)
            Toast.makeText(c, msg, Toast.LENGTH_LONG).show()
            return
        }
        if (DrawCore.busy) {
            Toast.makeText(c, c.getString(R.string.s_428), Toast.LENGTH_LONG).show()
            return
        }
        val prompt = promptEdit.text.toString().trim()
        if (prompt.isEmpty()) {
            Toast.makeText(c, c.getString(R.string.s_047), Toast.LENGTH_SHORT).show()
            return
        }
        val requestedSteps = stepsEdit.text.toString().toIntOrNull()?.coerceIn(1, 150) ?: 20
        denoisingStarted = false
        setGenerating(true)
        progressText.visibility = View.VISIBLE
        progressText.text = c.getString(R.string.s_126)
        curStep = 0
        totalStep = 0
        progressBar.progress = 0
        progressBar.visibility = View.VISIBLE
        startPreview()
        onStatus?.invoke(c.getString(R.string.s_165), false)
        val startedAt = System.currentTimeMillis()
        scope.launch {
            // 逐秒报“已耗时”，否则底层不报中间进度，看着像卡死
            val ticker = launch {
                while (true) {
                    delay(1000)
                    val sec = (System.currentTimeMillis() - startedAt) / 1000
                    progressText.post {
                        // Hearth reports denoising progress through the sampler callback. Keep the
                        // UI in Loading model until a callback matches this request's sample-step count.
                        val phase = if (denoisingStarted) c.getString(R.string.s_129) else c.getString(R.string.s_127)
                        val displayStep = if (previewFrameSeen && totalStep > 0) {
                            previewFrameStep.coerceIn(0, totalStep)
                        } else {
                            curStep.coerceAtLeast(0)
                        }
                        val stepInfo = if (totalStep > 0) c.getString(R.string.v_049, displayStep, totalStep) else ""
                        progressText.text = c.getString(R.string.v_050, phase, stepInfo, sec)
                        if (::procStepText.isInitialized) {
                            val processPhase = if (previewFrameSeen) c.getString(R.string.s_129) else phase
                            procStepText.text = c.getString(R.string.v_050, processPhase, stepInfo, sec)
                        }
                        if (totalStep > 0) {
                            val progress = (displayStep * 100 / totalStep).coerceIn(0, 100)
                            progressBar.progress = progress
                            if (::procProgressBar.isInitialized) {
                                procProgressBar.progress = progress
                            }
                        }
                    }
                }
            }
            try {
                val img = generateImage(
                    prompt,
                    onProgress = { cur, total ->
                        // ModelLoader::pretty_bytes_progress shares the native callback and can
                        // report tensor counts (e.g. 686/686), not denoising steps. Ignore those
                        // until the sampler reports a plausible step count for this request.
                        if (total in 1..(requestedSteps + 2)) {
                            denoisingStarted = true
                            curStep = cur.coerceIn(0, total)
                            totalStep = total
                        } else if (!denoisingStarted) {
                            curStep = 0
                            totalStep = 0
                        }
                    },
                    onPreview = { st, w, h, rgb -> onPreviewFrame(st, w, h, rgb) },
                )
                ticker.cancel()
                val bmp = toBitmap(img)
                val sec = (System.currentTimeMillis() - startedAt) / 1000
                resultImg.post {
                    DrawHistory.add(img)          // 先进历史（内存）
                    lastImage = bmp
                    resultImg.setImageBitmap(bmp)
                    resultInfo.text = c.getString(R.string.v_051, (img.width), (img.height), (img.seed), (sec))
                    refreshHistory()
                    switchPane(toResult = true)   // 出图后自动跳到结果页
                }
                val doneTxt = c.getString(R.string.v_052, (img.width), (img.height), (img.seed), (sec))
                progressText.post { progressText.text = doneTxt }
                if (::procStepText.isInitialized) procStepText.post { procStepText.text = doneTxt }
                onStatus?.invoke(c.getString(R.string.v_053, (sec)), false)
            } catch (e: Throwable) {
                ticker.cancel()
                val sec = (System.currentTimeMillis() - startedAt) / 1000
                if (cancelRequested) {
                    val stopTxt = c.getString(R.string.v_054, (sec))
                    progressText.post { progressText.text = stopTxt }
                    if (::procStepText.isInitialized) procStepText.post { procStepText.text = stopTxt }
                    onStatus?.invoke(c.getString(R.string.s_154), false)
                } else {
                    val errTxt = c.getString(R.string.v_059, (sec), (e.message ?: e.javaClass.simpleName)) + "\n" + c.getString(R.string.v_060)
                    progressText.post { progressText.text = errTxt }
                    if (::procStepText.isInitialized) procStepText.post { procStepText.text = errTxt }
                    onStatus?.invoke(c.getString(R.string.v_055, (e.message ?: e.javaClass.simpleName)), true)
                }
            } finally {
                setGenerating(false)
                progressBar.post { progressBar.visibility = View.GONE }
                switchResultPage(0)
            }
        }
    }

    @Volatile private var curStep = 0
    @Volatile private var totalStep = 0
    @Volatile private var denoisingStarted = false

    /** 用户是否已请求中断当前生成（用于把“取消”与“真失败”区分开）。 */
    @Volatile private var cancelRequested = false

    /** 供对话页复用的生成入口：把输入当正面提示词，其余参数取绘图页当前设置。 */
    suspend fun generateImage(
        prompt: String,
        onProgress: (Int, Int) -> Unit = { _, _ -> },
        onPreview: ((Int, Int, Int, ByteArray) -> Unit)? = null,
    ): ImageData {
        // 绘图接口正在出图时别插队：同一个 native 句柄同时跑两个任务会错乱
        if (DrawCore.busy) throw IllegalStateException(c.getString(R.string.s_428))
        val steps = stepsEdit.text.toString().toIntOrNull()?.coerceIn(1, 150) ?: 20
        val cfg = cfgEdit.text.toString().toFloatOrNull()?.coerceIn(1f, 30f) ?: 7.0f
        val eta = parseEta(etaEdit)
        val clipSkip = clipSkipSeekBar.progress + 1
        val seed = seedEdit.text.toString().toLongOrNull() ?: -1L
        val useSeed = if (seed < 0) System.currentTimeMillis() else seed
        // Keep dimensions divisible by 8 and clamp them to a practical range.
        val w = (widthEdit.text.toString().toIntOrNull() ?: 512).let { (it / 64) * 64 }.coerceIn(64, 2048)
        val hgt = (heightEdit.text.toString().toIntOrNull() ?: 512).let { (it / 64) * 64 }.coerceIn(64, 2048)
        onProgress(0, steps)
        cancelRequested = false

        val h = sdHandle
        if (h == 0L) throw IllegalStateException(c.getString(R.string.s_163))
        if (mainModel == null) throw IllegalStateException(c.getString(R.string.s_121))

        // Pass each selected LoRA with its own strength to Hearth.
        val loras = selectedLoras("t2i")
        val hires = currentHiresSettings()
        val adetailer = currentADetailerSettings()
        // EditText 必须在主线程读，先取出来再进 IO
        val negative = negEdit.text.toString()

        // 采样器/调度器：默认「自动」, 有任意 LoRA 时走 LCM。
        val sampler = pickSampler(loras.isNotEmpty())
        val scheduler = pickScheduler(loras.isNotEmpty())

        // Reserve the exact handle atomically with API generation and model replacement.
        if (!DrawCore.tryBeginUiGeneration(h)) {
            throw IllegalStateException(c.getString(R.string.s_428))
        }
        try {
            // 推理是同步阻塞的 native 调用（一张几百秒），绝不能跟调用方同线程——
            // 调用方基本都在 Dispatchers.Main，否则整个 UI 会卡死到出图为止。
            val bmp = withContext(Dispatchers.IO) {
                HearthEngine.render(
                    handle = h,
                    prompt = prompt,
                    negative = negative,
                    loras = loras,
                    width = w,
                    height = hgt,
                    steps = steps,
                    cfg = cfg,
                    clipSkip = clipSkip,
                    seed = useSeed,
                    sampler = sampler,
                    scheduler = scheduler,
                    eta = eta,
                    hires = hires,
                    adetailer = adetailer,
                    cb = { cur, total -> onProgress(cur, total) },
                    onPreview = onPreview?.let { f ->
                        HearthEngine.PreviewCallback { st, w, h, rgb -> f(st, w, h, rgb) }
                    },
                )
            } ?: throw IllegalStateException(c.getString(R.string.s_147))
            onProgress(steps, steps)
            return ImageData(bmp, useSeed)
        } finally {
            DrawCore.endUiGeneration()
        }
    }

    /** 采样器选择：索引 0 = 自动；其余按 HearthEngine.Sampler 顺序 */
    private fun pickSampler(hasLora: Boolean): HearthEngine.Sampler {
        val idx = samplerSpinner.selectedItemPosition
        if (idx <= 0) return HearthEngine.Sampler.EULER_A
        return HearthEngine.Sampler.HEARTH_SUPPORTED.getOrElse(idx - 1) { HearthEngine.Sampler.EULER_A }
    }

    /** -1 keeps the sampler-specific native ETA default; explicit values are clamped to 0..1. */
    private fun parseEta(edit: EditText): Float {
        val value = edit.text.toString().trim().toFloatOrNull() ?: return -1.0f
        return if (value < 0.0f) -1.0f else value.coerceIn(0.0f, 1.0f)
    }

    /** 调度器选择：索引 0 = 自动 */
    private fun pickScheduler(hasLora: Boolean): HearthEngine.Scheduler {
        val idx = schedulerSpinner.selectedItemPosition
        if (idx <= 0) return HearthEngine.Scheduler.DISCRETE
        return HearthEngine.Scheduler.HEARTH_SUPPORTED.getOrElse(idx - 1) { HearthEngine.Scheduler.DISCRETE }
    }

    // ================= 图生图生成 =================

    /** 图生图采样器（对应图生图页的 spinner） */
    private fun pickI2iSampler(hasLora: Boolean): HearthEngine.Sampler {
        val idx = i2iSamplerSpinner.selectedItemPosition
        if (idx <= 0) return HearthEngine.Sampler.EULER_A
        return HearthEngine.Sampler.HEARTH_SUPPORTED.getOrElse(idx - 1) { HearthEngine.Sampler.EULER_A }
    }

    /** 图生图调度器 */
    private fun pickI2iScheduler(hasLora: Boolean): HearthEngine.Scheduler {
        val idx = i2iSchedulerSpinner.selectedItemPosition
        if (idx <= 0) return HearthEngine.Scheduler.DISCRETE
        return HearthEngine.Scheduler.HEARTH_SUPPORTED.getOrElse(idx - 1) { HearthEngine.Scheduler.DISCRETE }
    }

    /** 点图生图页的「生成」按钮 */
    private fun generateFromUiI2i() {
        persistAll()
        if (!isReady()) {
            val msg = if (llmLoaded) c.getString(R.string.s_100)
            else c.getString(R.string.s_173)
            Toast.makeText(c, msg, Toast.LENGTH_LONG).show()
            return
        }
        if (DrawCore.busy) {
            Toast.makeText(c, c.getString(R.string.s_428), Toast.LENGTH_LONG).show()
            return
        }
        if (i2iBitmap == null) {
            Toast.makeText(c, c.getString(R.string.s_214), Toast.LENGTH_SHORT).show()
            return
        }
        val prompt = i2iPromptEdit.text.toString().trim()
        if (prompt.isEmpty()) {
            Toast.makeText(c, c.getString(R.string.s_047), Toast.LENGTH_SHORT).show()
            return
        }
        val requestedSteps = i2iStepsEdit.text.toString().toIntOrNull()?.coerceIn(1, 150) ?: 20
        denoisingStarted = false
        setGenerating(true)
        i2iProgressText.visibility = View.VISIBLE
        i2iProgressText.text = c.getString(R.string.s_126)
        curStep = 0
        totalStep = 0
        i2iProgressBar.progress = 0
        i2iProgressBar.visibility = View.VISIBLE
        startPreview()
        onStatus?.invoke(c.getString(R.string.s_165), false)
        val startedAt = System.currentTimeMillis()
        scope.launch {
            val ticker = launch {
                while (true) {
                    delay(1000)
                    val sec = (System.currentTimeMillis() - startedAt) / 1000
                    i2iProgressText.post {
                        val phase = if (denoisingStarted) c.getString(R.string.s_129) else c.getString(R.string.s_127)
                        val displayStep = if (previewFrameSeen && totalStep > 0) {
                            previewFrameStep.coerceIn(0, totalStep)
                        } else {
                            curStep.coerceAtLeast(0)
                        }
                        val stepInfo = if (totalStep > 0) c.getString(R.string.v_049, displayStep, totalStep) else ""
                        i2iProgressText.text = c.getString(R.string.v_050, phase, stepInfo, sec)
                        if (::procStepText.isInitialized) {
                            val processPhase = if (previewFrameSeen) c.getString(R.string.s_129) else phase
                            procStepText.text = c.getString(R.string.v_050, processPhase, stepInfo, sec)
                        }
                        if (totalStep > 0) {
                            val progress = (displayStep * 100 / totalStep).coerceIn(0, 100)
                            i2iProgressBar.progress = progress
                            if (::procProgressBar.isInitialized) {
                                procProgressBar.progress = progress
                            }
                        }
                    }
                }
            }
            try {
                val img = generateImageI2i(
                    prompt,
                    onProgress = { cur, total ->
                        if (total in 1..(requestedSteps + 2)) {
                            denoisingStarted = true
                            curStep = cur.coerceIn(0, total)
                            totalStep = total
                        } else if (!denoisingStarted) {
                            curStep = 0
                            totalStep = 0
                        }
                    },
                    onPreview = { st, w, h, rgb -> onPreviewFrame(st, w, h, rgb) },
                )
                ticker.cancel()
                val bmp = toBitmap(img)
                val sec = (System.currentTimeMillis() - startedAt) / 1000
                resultImg.post {
                    DrawHistory.add(img)
                    lastImage = bmp
                    resultImg.setImageBitmap(bmp)
                    resultInfo.text = c.getString(R.string.v_051, (img.width), (img.height), (img.seed), (sec))
                    refreshHistory()
                    switchPane(toResult = true)
                }
                val i2iDoneTxt = c.getString(R.string.v_052, (img.width), (img.height), (img.seed), (sec))
                i2iProgressText.post { i2iProgressText.text = i2iDoneTxt }
                if (::procStepText.isInitialized) procStepText.post { procStepText.text = i2iDoneTxt }
                onStatus?.invoke(c.getString(R.string.v_053, (sec)), false)
            } catch (e: Throwable) {
                ticker.cancel()
                val sec = (System.currentTimeMillis() - startedAt) / 1000
                if (cancelRequested) {
                    val i2iStopTxt = c.getString(R.string.v_054, (sec))
                    i2iProgressText.post { i2iProgressText.text = i2iStopTxt }
                    if (::procStepText.isInitialized) procStepText.post { procStepText.text = i2iStopTxt }
                    onStatus?.invoke(c.getString(R.string.s_154), false)
                } else {
                    val i2iErrTxt = c.getString(R.string.v_059, (sec), (e.message ?: e.javaClass.simpleName)) + "\n" + c.getString(R.string.v_060)
                    i2iProgressText.post { i2iProgressText.text = i2iErrTxt }
                    if (::procStepText.isInitialized) procStepText.post { procStepText.text = i2iErrTxt }
                    onStatus?.invoke(c.getString(R.string.v_055, (e.message ?: e.javaClass.simpleName)), true)
                }
            } finally {
                setGenerating(false)
                i2iProgressBar.post { i2iProgressBar.visibility = View.GONE }
                switchResultPage(0)
            }
        }
    }

    /** The Hearth adapter has no img2img pipeline; fail explicitly instead of silently running txt2img. */
    private suspend fun generateImageI2i(
        @Suppress("UNUSED_PARAMETER") prompt: String,
        @Suppress("UNUSED_PARAMETER") onProgress: (Int, Int) -> Unit = { _, _ -> },
        @Suppress("UNUSED_PARAMETER") onPreview: ((Int, Int, Int, ByteArray) -> Unit)? = null,
    ): ImageData = throw IllegalStateException(c.getString(R.string.draw_hearth_i2i_unavailable))

        fun toBitmap(img: ImageData): Bitmap = img.bitmap

    /** 供对话页调用：把生成过程的预览帧转给「过程」页 */
    fun previewFrame(step: Int, w: Int, h: Int, rgb: ByteArray) = onPreviewFrame(step, w, h, rgb)

    /** 点「中断生成」：先给个即时反馈，再请求 native 中断 */
    private fun doCancel() {
        cancel()
        progressText.text = c.getString(R.string.s_125)
    }

    /** 统一切换按钮的「开始生成 / 中断生成」外观与行为 */
    private fun setGenerating(g: Boolean) {
        generating = g
        genBtn.post {
            genBtn.isEnabled = true
            genBtn.text = if (g) c.getString(R.string.s_040) else c.getString(R.string.s_098)
            genBtn.setBackgroundColor(if (g) pal.stop else primary)
        }
        if (::i2iGenBtn.isInitialized) i2iGenBtn.post {
            i2iGenBtn.isEnabled = true
            i2iGenBtn.text = if (g) c.getString(R.string.s_040) else c.getString(R.string.s_098)
            i2iGenBtn.setBackgroundColor(if (g) pal.stop else primary)
        }
    }

    fun cancel() {
        cancelRequested = true
        if (sdHandle != 0L) runCatching { HearthEngine.nativeCancel(sdHandle) }
    }

    fun onActivityResult(requestCode: Int, resultCode: Int, data: android.content.Intent?): Boolean {
        if (handleLoraFusionPickedFile(requestCode, resultCode, data)) return true
        if (handleHiresUpscalerPickedFile(requestCode, resultCode, data)) return true
        if (handleADetailerPickedFile(requestCode, resultCode, data)) return true
        if (handleResAdapterPickedFile(requestCode, resultCode, data)) return true
        if (requestCode != REQ_IMAGE) return false
        if (resultCode != Activity.RESULT_OK) return true
        val uri = data?.data ?: return true
        scope.launch {
            val bmp = withContext(Dispatchers.IO) { decodePickedImage(uri) }
            if (bmp == null) toast(c.getString(R.string.s_215)) else applyPickedImage(bmp)
        }
        return true
    }

    fun release() {
        DrawCore.release()   // 顺序不能反：先断开 API 侧的引用，再真正释放 native 句柄
        if (sdHandle != 0L) runCatching { HearthEngine.nativeFree(sdHandle) }
        sdHandle = 0L
    }

    // ================= UI 小工具 =================

    private fun toast(t: String) = Toast.makeText(c, t, Toast.LENGTH_SHORT).show()

    private fun dp(v: Int): Int = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), c.resources.displayMetrics
    ).toInt()

    private fun card() = LinearLayout(c).apply {
        orientation = LinearLayout.VERTICAL
        setBackgroundColor(pal.surface)
        setPadding(dp(14), dp(12), dp(14), dp(12))
        layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = dp(10) }
    }

    private fun title(t: String) = TextView(c).apply {
        text = t
        textSize = 14.5f
        setTextColor(textColor)
        typeface = Typeface.DEFAULT_BOLD
    }

    private fun body(t: String) = TextView(c).apply {
        text = t
        textSize = 12.5f
        setTextColor(subText)
        setPadding(0, dp(6), 0, 0)
    }

    private fun smallLabel(t: String) = TextView(c).apply {
        text = t
        textSize = 11.5f
        setTextColor(subText)
        setPadding(0, dp(8), 0, 0)
    }

    private fun labeledEdit(hint: String, singleLine: Boolean, minLines: Int = 1) = EditText(c).apply {
        this.hint = hint
        textSize = 13f
        setTextColor(textColor)
        setHintTextColor(pal.hint)
        setBackgroundColor(pal.fieldBg)
        setPadding(dp(10), dp(8), dp(10), dp(8))
        isSingleLine = singleLine
        if (!singleLine) {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            this.minLines = minLines
        }
    }

    private fun smallNumber(def: String) = EditText(c).apply {
        setText(def)
        textSize = 13f
        setTextColor(textColor)
        inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL or InputType.TYPE_NUMBER_FLAG_SIGNED
        setBackgroundColor(pal.fieldBg)
        setPadding(dp(10), dp(8), dp(10), dp(8))
    }

    /** CLIP skip slider: 1 keeps the standard final-layer output; higher values skip more CLIP layers. */
    private fun buildClipSkipControl(
        preferenceKey: String,
        initialValue: Int,
    ): Pair<View, android.widget.SeekBar> {
        val column = LinearLayout(c).apply { orientation = LinearLayout.VERTICAL }
        val valueLabel = TextView(c).apply {
            textSize = 12f
            setTextColor(textColor)
        }
        val slider = android.widget.SeekBar(c).apply {
            max = 11
            progress = initialValue.coerceIn(1, 12) - 1
            contentDescription = c.getString(R.string.draw_clip_skip_label)
        }
        fun updateLabel(value: Int) {
            valueLabel.text = c.getString(R.string.draw_clip_skip_value, value)
        }
        updateLabel(slider.progress + 1)
        slider.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(
                seekBar: android.widget.SeekBar?,
                progress: Int,
                fromUser: Boolean,
            ) {
                val value = progress + 1
                updateLabel(value)
                if (fromUser) prefs().edit().putInt(preferenceKey, value).apply()
            }

            override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {}
        })
        column.addView(valueLabel)
        column.addView(slider, matchWrap(top = 1))
        column.addView(TextView(c).apply {
            text = c.getString(R.string.draw_clip_skip_hint)
            textSize = 10.5f
            setTextColor(subText)
            setPadding(0, dp(2), 0, 0)
        })
        return column to slider
    }

    private fun paramRow(label: String, input: View, hint: String): View {
        val wrap = LinearLayout(c).apply { orientation = LinearLayout.VERTICAL }
        wrap.addView(smallLabel(label))
        wrap.addView(input, matchWrap(top = 4))
        wrap.addView(TextView(c).apply {
            text = hint
            textSize = 10.5f
            setTextColor(subText)
            setPadding(0, dp(3), 0, 0)
        })
        return wrap
    }

    private fun matchWrap(top: Int = 0, bottom: Int = 0) = LinearLayout.LayoutParams(
        LinearLayout.LayoutParams.MATCH_PARENT,
        LinearLayout.LayoutParams.WRAP_CONTENT
    ).apply {
        topMargin = dp(top)
        bottomMargin = dp(bottom)
    }

    /** 顶部菜单的一项 */
    private fun tabItem(t: String) = TextView(c).apply {
        text = t
        textSize = 13.5f
        gravity = Gravity.CENTER
        setPadding(0, dp(11), 0, dp(11))
    }

    /** 切换「参数 / 结果」面板（带动画翻页），并高亮当前项 */
    private fun switchPane(toResult: Boolean) {
        if (!::paramPane.isInitialized) return
        if (::pager.isInitialized) pager.setCurrentItem(if (toResult) 1 else 0, true)
        highlightTabs(toResult)
    }

    /** 同步顶部菜单的高亮；手势翻页时也由它更新 */
    private fun highlightTabs(toResult: Boolean) {
        if (!::tabParams.isInitialized) return
        for ((tv, sel) in listOf(tabParams to !toResult, tabResult to toResult)) {
            tv.setTextColor(if (sel) primary else subText)
            tv.typeface = if (sel) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
            tv.setBackgroundColor(if (sel) pal.primarySoft else pal.surface)
        }
    }

    /** 结果面板内切换「结果 / 过程」子页 */
    private fun switchResultPage(p: Int) {
        if (!::resultContent.isInitialized) return
        resultPage = p
        resultContent.visibility = if (p == 0) View.VISIBLE else View.GONE
        processContent.visibility = if (p == 1) View.VISIBLE else View.GONE
        for ((tv, sel) in listOf(tabResResult to (p == 0), tabResProcess to (p == 1))) {
            tv.setTextColor(if (sel) primary else subText)
            tv.typeface = if (sel) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
            tv.setBackgroundColor(if (sel) pal.primarySoft else pal.surface)
        }
    }

    /** 开始生成：清掉上一张预览并切到「过程」页 */
    private fun startPreview() {
        if (!::processContent.isInitialized) return
        previewFrameSeen = false
        previewFrameStep = 0
        synchronized(processPreviewLock) { pendingProcessPreview = null }
        previewImg.setImageDrawable(null)
        procProgressBar.progress = 0
        procStepText.text = c.getString(R.string.s_126)
        switchPane(toResult = true)
        switchResultPage(1)
    }

    /** Render the newest pending preview and update its matching step/progress together. */
    private fun onPreviewFrame(step: Int, w: Int, h: Int, rgb: ByteArray) {
        if (!::previewImg.isInitialized) return
        denoisingStarted = true
        val bmp = HearthEngine.rgbToBitmap(w, h, rgb) ?: return
        val frame = ProcessPreviewFrame(kotlin.math.abs(step), bmp)
        var post = false
        synchronized(processPreviewLock) {
            pendingProcessPreview = frame
            if (!processPreviewPostQueued) {
                processPreviewPostQueued = true
                post = true
            }
        }
        if (post) previewImg.post { drainProcessPreview() }
    }

    private fun drainProcessPreview() {
        val frame = synchronized(processPreviewLock) {
            val next = pendingProcessPreview
            pendingProcessPreview = null
            processPreviewPostQueued = false
            next
        }

        if (frame != null) {
            previewImg.setImageBitmap(frame.bitmap)
            val total = totalStep
            previewFrameStep = if (total > 0) frame.step.coerceIn(0, total) else frame.step
            previewFrameSeen = true
            if (total > 0) {
                val progress = (previewFrameStep * 100 / total).coerceIn(0, 100)
                procProgressBar.progress = progress
                if (::progressBar.isInitialized) progressBar.progress = progress
                if (::i2iProgressBar.isInitialized) i2iProgressBar.progress = progress
                procStepText.text = c.getString(R.string.s_129) +
                    c.getString(R.string.v_049, previewFrameStep, total)
            }
        }

        var post = false
        synchronized(processPreviewLock) {
            if (pendingProcessPreview != null && !processPreviewPostQueued) {
                processPreviewPostQueued = true
                post = true
            }
        }
        if (post) previewImg.post { drainProcessPreview() }
    }

    /** 让某个面板回到最顶处（等布局完成后执行，避免刚切页时高度还是 0） */
    private fun scrollTop(s: ScrollView?) {
        s ?: return
        s.post { s.smoothScrollTo(0, 0) }
    }

    /** 给面板包一层可纵向滚动的 ScrollView（翻页控件需要一个确定高度的子项） */
    private fun scrollWrap(inner: View) = ScrollView(c).apply {
        isFillViewport = true
        addView(inner, ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
    }

    /** ViewPager 用的最小 PagerAdapter：直接复用已建好的两个面板，不重建 */
    private class PaneAdapter(private val panes: List<View>) : PagerAdapter() {
        override fun getCount() = panes.size
        override fun isViewFromObject(view: View, obj: Any) = view === obj
        override fun instantiateItem(container: ViewGroup, position: Int): Any {
            val v = panes[position]
            (v.parent as? ViewGroup)?.removeView(v)
            container.addView(v)
            return v
        }
        override fun destroyItem(container: ViewGroup, position: Int, obj: Any) {
            (obj as? View)?.let { container.removeView(it) }
        }
    }

    /** 宽 × 高 输入行 */
    private fun whRow(label: String, hint: String, wEdit: EditText, hEdit: EditText): View {
        val wrap = LinearLayout(c).apply { orientation = LinearLayout.VERTICAL }
        wrap.addView(smallLabel(label))
        val row = LinearLayout(c).apply { orientation = LinearLayout.HORIZONTAL }
        row.addView(wEdit, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        row.addView(TextView(c).apply {
            text = " × "
            textSize = 14f
            setTextColor(textColor)
            gravity = Gravity.CENTER
        }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        row.addView(hEdit, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        wrap.addView(row, matchWrap(top = 4))
        wrap.addView(TextView(c).apply {
            text = hint
            textSize = 10.5f
            setTextColor(subText)
            setPadding(0, dp(3), 0, 0)
        })
        return wrap
    }

    private fun choiceSpinner(items: List<String>) = android.widget.Spinner(c).apply {
        adapter = android.widget.ArrayAdapter(c, android.R.layout.simple_spinner_dropdown_item, items)
    }

    // ---- 参数持久化（提示词不保存，每次重来） ----

    private fun prefs() = c.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private fun loadParam(key: String, def: String): String = prefs().getString(key, def) ?: def

    private fun loadInt(key: String, def: Int): Int = prefs().getInt(key, def)

    /**
     * 参数改动即时落盘。绑定时机在 setText(初值) 之后，
     * 所以初始化不会把恢复出来的值又写一遍。
     */
    private fun bindParam(edit: EditText, key: String) {
        edit.addTextChangedListener(object : android.text.TextWatcher {
            override fun afterTextChanged(s: android.text.Editable?) {
                val v = s?.toString().orEmpty()
                if (v.isNotBlank()) prefs().edit().putString(key, v).apply()
                if (key == KEY_LORA_SCALE || key.startsWith(KEY_LORA_SCALE + "_slot")) {
                    syncDrawCore()
                }
            }
            override fun beforeTextChanged(s: CharSequence?, st: Int, b: Int, cnt: Int) {}
            override fun onTextChanged(s: CharSequence?, st: Int, b: Int, cnt: Int) {}
        })
    }

    private fun bindOptionalParam(edit: EditText, key: String) {
        edit.addTextChangedListener(object : android.text.TextWatcher {
            override fun afterTextChanged(s: android.text.Editable?) {
                if (s.isNullOrEmpty()) prefs().edit().remove(key).apply()
            }
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        })
    }

    /**
     * 把当前所有参数写盘 —— 由 MainActivity.onPause() 与「开始生成」兜底调用。
     * 不能只靠输入框的 TextWatcher：切 tab、直接杀进程等场景可能漏掉。
     */
    fun persistAll() {
        if (!::stepsEdit.isInitialized) return
        val e = prefs().edit()
        fun put(k: String, ed: EditText) {
            val v = ed.text.toString()
            if (v.isNotBlank()) e.putString(k, v)
        }
        put(KEY_W, widthEdit)
        put(KEY_H, heightEdit)
        put(KEY_STEPS, stepsEdit)
        put(KEY_CFG, cfgEdit)
        put(KEY_ETA, etaEdit)
        e.putInt(KEY_CLIP_SKIP, clipSkipSeekBar.progress + 1)
        put(KEY_SEED, seedEdit)
        t2iLoraWeights.forEachIndexed { slot, edit ->
            put(loraWeightKey("t2i", slot), edit)
        }
        t2iLoraSpinners.forEachIndexed { slot, spinner ->
            val name = (spinner.selectedItem as? String)
                ?.takeUnless { it == LORA_SLOT_NONE }.orEmpty()
            e.putString(loraSlotKey("t2i", slot), name)
        }
        e.putInt(KEY_SAMPLER, samplerSpinner.selectedItemPosition)
        e.putInt(KEY_SCHEDULER, schedulerSpinner.selectedItemPosition)
        if (::i2iStepsEdit.isInitialized) {
            put(KEY_I2I_W, i2iWidthEdit)
            put(KEY_I2I_H, i2iHeightEdit)
            put(KEY_I2I_STEPS, i2iStepsEdit)
            put(KEY_I2I_CFG, i2iCfgEdit)
            put(KEY_I2I_ETA, i2iEtaEdit)
            e.putInt(KEY_I2I_CLIP_SKIP, i2iClipSkipSeekBar.progress + 1)
            put(KEY_I2I_SEED, i2iSeedEdit)
            put(KEY_I2I_STRENGTH, i2iStrengthEdit)
            i2iLoraWeights.forEachIndexed { slot, edit ->
                put(loraWeightKey("i2i", slot), edit)
            }
            i2iLoraSpinners.forEachIndexed { slot, spinner ->
                val name = (spinner.selectedItem as? String)
                    ?.takeUnless { it == LORA_SLOT_NONE }.orEmpty()
                e.putString(loraSlotKey("i2i", slot), name)
            }
            e.putInt(KEY_I2I_SAMPLER, i2iSamplerSpinner.selectedItemPosition)
            e.putInt(KEY_I2I_SCHEDULER, i2iSchedulerSpinner.selectedItemPosition)
        }
        e.apply()
        syncDrawCore()   // 参数一改就同步给 API 侧（下一次请求即生效）
    }

    private fun samplerAt(spinner: android.widget.Spinner): HearthEngine.Sampler? {
        val position = spinner.selectedItemPosition
        return if (position <= 0) null else HearthEngine.Sampler.HEARTH_SUPPORTED.getOrNull(position - 1)
    }

    private fun karrasSchedulerPosition(): Int = HearthEngine.Scheduler.HEARTH_SUPPORTED.indexOf(HearthEngine.Scheduler.KARRAS) + 1

    private fun tcdSchedulerPosition(): Int = -1

    private fun tcdSamplerPosition(): Int = -1

    /** Hearth exposes samplers and sigma schedules as separate choices. */
    private fun syncTcdPresetOnRestore(
        sampler: android.widget.Spinner,
        scheduler: android.widget.Spinner,
        samplerKey: String,
        schedulerKey: String,
    ) {
        val isTcd = scheduler.selectedItemPosition == tcdSchedulerPosition()
        if (isTcd) {
            val position = tcdSamplerPosition()
            sampler.setSelection(position, false)
            prefs().edit().putInt(samplerKey, position).putInt(schedulerKey, tcdSchedulerPosition()).apply()
        }
        sampler.isEnabled = !isTcd
    }

    /** Restore the explicit "DPM++ SDE Karras" preset coherently with its scheduler. */
    private fun syncKarrasPresetOnRestore(
        sampler: android.widget.Spinner,
        scheduler: android.widget.Spinner,
        schedulerKey: String,
    ) {
        if (samplerAt(sampler) == HearthEngine.Sampler.DPMPP_SDE_KARRAS) {
            val position = karrasSchedulerPosition()
            scheduler.setSelection(position, false)
            prefs().edit().putInt(schedulerKey, position).apply()
        }
    }

    private fun samplerPresetListener(
        key: String,
        sampler: android.widget.Spinner,
        scheduler: android.widget.Spinner,
    ) = object : android.widget.AdapterView.OnItemSelectedListener {
        override fun onItemSelected(p: android.widget.AdapterView<*>?, v: View?, pos: Int, id: Long) {
            prefs().edit().putInt(key, pos).apply()
            if (samplerAt(sampler) == HearthEngine.Sampler.DPMPP_SDE_KARRAS) {
                val karrasPosition = karrasSchedulerPosition()
                if (scheduler.selectedItemPosition != karrasPosition) {
                    scheduler.setSelection(karrasPosition)
                }
            }
        }

        override fun onNothingSelected(p: android.widget.AdapterView<*>?) {}
    }

    private fun schedulerPresetListener(
        key: String,
        samplerKey: String,
        sampler: android.widget.Spinner,
    ) = object : android.widget.AdapterView.OnItemSelectedListener {
        override fun onItemSelected(p: android.widget.AdapterView<*>?, v: View?, pos: Int, id: Long) {
            prefs().edit().putInt(key, pos).apply()
            val isTcdPosition = pos == tcdSchedulerPosition()
            sampler.isEnabled = !isTcdPosition
            if (isTcdPosition) {
                val tcdPosition = tcdSamplerPosition()
                sampler.setSelection(tcdPosition)
                prefs().edit().putInt(samplerKey, tcdPosition).apply()
            }
            val current = samplerAt(sampler)
            val isKarrasPosition = pos == karrasSchedulerPosition()
            if (current == HearthEngine.Sampler.DPMPP_SDE_KARRAS && !isKarrasPosition) {
                val standardSdePosition = HearthEngine.Sampler.HEARTH_SUPPORTED.indexOf(HearthEngine.Sampler.DPMPP_SDE) + 1
                if (standardSdePosition > 0 && sampler.selectedItemPosition != standardSdePosition) {
                    sampler.setSelection(standardSdePosition)
                }
            }
        }

        override fun onNothingSelected(p: android.widget.AdapterView<*>?) {}
    }

    /** Spinner 选择落盘 */
    private fun persistSpinner(key: String) = object : android.widget.AdapterView.OnItemSelectedListener {
        override fun onItemSelected(p: android.widget.AdapterView<*>?, v: View?, pos: Int, id: Long) {
            prefs().edit().putInt(key, pos).apply()
        }
        override fun onNothingSelected(p: android.widget.AdapterView<*>?) {}
    }

    // ---- 默认值（文生图 / 图生图按 LoRA 开/关各一套） ----

    /** 当前 LoRA 状态（0=关 1=开），作默认值键后缀 */
    private fun loraTag(): String = if (useLora) "1" else "0"

    private fun builtinSteps(): String = if (useLora) "6" else "20"
    private fun builtinCfg(): String = if (useLora) "1.8" else "7.0"

    /** 取某页参数的默认值：优先用户保存的默认，否则用内置 */
    private fun defOf(page: String, name: String, builtin: String): String =
        prefs().getString("def_${page}_${name}_${loraTag()}", null) ?: builtin

    private fun defOfInt(page: String, name: String, builtin: Int): Int =
        prefs().getInt("def_${page}_${name}_${loraTag()}", builtin)

    /** 把当前页当前参数存为该页（当前 LoRA 状态）的默认值 */
    fun saveDefaults(page: String) {
        val lg = loraTag()
        val e = prefs().edit()
        fun put(n: String, ed: EditText) { e.putString("def_${page}_${n}_$lg", ed.text.toString()) }
        when (page) {
            "t2i" -> {
                put("w", widthEdit); put("h", heightEdit); put("steps", stepsEdit)
                put("cfg", cfgEdit); put("eta", etaEdit); put("seed", seedEdit)
                e.putInt("def_t2i_clipSkip_$lg", clipSkipSeekBar.progress + 1)
                t2iLoraWeights.forEachIndexed { slot, edit ->
                    put(if (slot == 0) "loraScale" else "loraScale" + (slot + 1), edit)
                }
                e.putInt("def_t2i_sampler_$lg", samplerSpinner.selectedItemPosition)
                e.putInt("def_t2i_scheduler_$lg", schedulerSpinner.selectedItemPosition)
            }
            "i2i" -> {
                put("strength", i2iStrengthEdit)
                put("w", i2iWidthEdit); put("h", i2iHeightEdit); put("steps", i2iStepsEdit)
                put("cfg", i2iCfgEdit); put("eta", i2iEtaEdit); put("seed", i2iSeedEdit)
                e.putInt("def_i2i_clipSkip_$lg", i2iClipSkipSeekBar.progress + 1)
                i2iLoraWeights.forEachIndexed { slot, edit ->
                    put(if (slot == 0) "loraScale" else "loraScale" + (slot + 1), edit)
                }
                e.putInt("def_i2i_sampler_$lg", i2iSamplerSpinner.selectedItemPosition)
                e.putInt("def_i2i_scheduler_$lg", i2iSchedulerSpinner.selectedItemPosition)
            }
        }
        e.apply()
        toast(c.getString(R.string.s_250))
    }

    /** 把默认值套用到指定页（当前 LoRA 状态） */
    fun applyDefaults(page: String) {
        when (page) {
            "t2i" -> {
                widthEdit.setText(defOf("t2i", "w", "512"))
                heightEdit.setText(defOf("t2i", "h", "512"))
                stepsEdit.setText(defOf("t2i", "steps", builtinSteps()))
                cfgEdit.setText(defOf("t2i", "cfg", builtinCfg()))
                etaEdit.setText(defOf("t2i", "eta", "-1"))
                clipSkipSeekBar.progress = defOfInt("t2i", "clipSkip", 1).coerceIn(1, 12) - 1
                seedEdit.setText(defOf("t2i", "seed", "-1"))
                t2iLoraWeights.forEachIndexed { slot, edit ->
                    edit.setText(defOf("t2i", if (slot == 0) "loraScale" else "loraScale" + (slot + 1), "1.0"))
                }
                samplerSpinner.setSelection(defOfInt("t2i", "sampler", 0), false)
                schedulerSpinner.setSelection(defOfInt("t2i", "scheduler", 0), false)
            }
            "i2i" -> {
                i2iStrengthEdit.setText(defOf("i2i", "strength", "0.75"))
                i2iWidthEdit.setText(defOf("i2i", "w", "512"))
                i2iHeightEdit.setText(defOf("i2i", "h", "512"))
                i2iStepsEdit.setText(defOf("i2i", "steps", builtinSteps()))
                i2iCfgEdit.setText(defOf("i2i", "cfg", builtinCfg()))
                i2iEtaEdit.setText(defOf("i2i", "eta", "-1"))
                i2iClipSkipSeekBar.progress = defOfInt("i2i", "clipSkip", 1).coerceIn(1, 12) - 1
                i2iSeedEdit.setText(defOf("i2i", "seed", "-1"))
                i2iLoraWeights.forEachIndexed { slot, edit ->
                    edit.setText(defOf("i2i", if (slot == 0) "loraScale" else "loraScale" + (slot + 1), "1.0"))
                }
                i2iSamplerSpinner.setSelection(defOfInt("i2i", "sampler", 0), false)
                i2iSchedulerSpinner.setSelection(defOfInt("i2i", "scheduler", 0), false)
            }
        }
    }

    /** 参数卡底部的「设为默认值 / 恢复默认」一行（点击后需确认） */
    private fun defaultButtonsRow(page: String): LinearLayout {
        val row = LinearLayout(c).apply { orientation = LinearLayout.HORIZONTAL }
        row.addView(ghostButton(c.getString(R.string.s_248)) { confirmDefaults(page, true) },
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        row.addView(ghostButton(c.getString(R.string.s_249)) { confirmDefaults(page, false) },
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                .apply { leftMargin = dp(8) })
        return row
    }

    /** 点击后先弹确认框，确认才真正生效（save=true 存为默认，false 恢复默认） */
    private fun confirmDefaults(page: String, save: Boolean) {
        val pageName = c.getString(when (page) {
            "t2i" -> R.string.s_207
            "i2i" -> R.string.s_208
            else -> R.string.s_221
        })
        android.app.AlertDialog.Builder(act)
            .setTitle(c.getString(if (save) R.string.s_248 else R.string.s_249))
            .setMessage(c.getString(if (save) R.string.s_256 else R.string.s_257, pageName))
            .setPositiveButton(c.getString(R.string.s_258)) { _, _ ->
                if (save) saveDefaults(page) else applyDefaults(page)
            }
            .setNegativeButton(c.getString(R.string.s_066), null)
            .show()
    }

    private fun ghostButton(t: String, onClick: () -> Unit) = Button(c).apply {
        text = t
        textSize = 12.5f
        setTextColor(primary)
        setBackgroundColor(pal.primarySoft)
        setOnClickListener { onClick() }
    }

    /** 结果图保存用的文件名 */
    private fun saveName(): String = "ponko_${System.currentTimeMillis()}.png"

    // ---- 生成历史（只活在内存里，进程结束即清空） ----

    /** 点缩略图：把大图切到这一张 */
    private fun showHistoryImage(img: ImageData) {
        lastImage = img.bitmap
        resultImg.setImageBitmap(img.bitmap)
        resultInfo.text = "${img.width}×${img.height} · seed ${img.seed}"
    }

    /** 重建缩略图列表 */
    private fun refreshHistory() {
        if (!::historyRow.isInitialized) return
        val items = DrawHistory.list()
        histEmpty.visibility = if (items.isEmpty()) View.VISIBLE else View.GONE
        historyRow.removeAllViews()
        for (img in items) {
            val iv = ImageView(c).apply {
                setImageBitmap(img.bitmap)
                scaleType = ImageView.ScaleType.CENTER_CROP
                setBackgroundColor(pal.mediaBg)
                setOnClickListener { showHistoryImage(img) }
                setOnLongClickListener {
                    android.app.AlertDialog.Builder(act)
                        .setMessage(c.getString(R.string.s_057))
                        .setPositiveButton(c.getString(R.string.s_050)) { _, _ -> DrawHistory.remove(img); refreshHistory() }
                        .setNegativeButton(c.getString(R.string.s_066), null)
                        .show()
                    true
                }
            }
            historyRow.addView(iv, LinearLayout.LayoutParams(dp(76), dp(76)).apply { rightMargin = dp(8) })
        }
    }
}


/** 生成结果：Bitmap + 尺寸 + 实际使用的种子 */
class ImageData(val bitmap: Bitmap, val seed: Long) {
    val width: Int get() = bitmap.width
    val height: Int get() = bitmap.height
}

/**
 * 绘图结果历史 —— 纯内存，App 进程结束即清空（用户要求「保留至程序被杀」）。
 * 放在 object 里而不是 DrawPage 字段里，这样 Activity 重建（旋转屏幕、
 * 内存回收后恢复）也能看到同一份历史。
 */
object DrawHistory {
    /** 最多留这么多张，防止一直生成把内存撑爆（512×512 一张约 1MB） */
    private const val MAX = 30
    private val items = mutableListOf<ImageData>()

    /** 新的排在最前 */
    fun add(img: ImageData) {
        items.add(0, img)
        while (items.size > MAX) items.removeAt(items.size - 1)
    }

    fun list(): List<ImageData> = items

    fun remove(img: ImageData) {
        items.remove(img)
    }

    fun clear() = items.clear()

    fun size() = items.size
}
