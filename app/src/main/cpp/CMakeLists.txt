# Ponko · (c) 2026 visitor257 · MIT License · github.com/visitor257/Ponko
# Ponko —— 自编 stable-diffusion.cpp 的构建脚本
#
# 这里做三件事：
#   1. 以子目录方式引入 sd.cpp（源码在 ./sd，含内联的 ggml，已剔除不用的词表/示例等）
#   2. 使用目标设备的编译标志（默认 Exynos 2400e: ARMv8.6-A + dotprod + I8MM + FP16 + -O3）
#   3. 编译 Rust-facing plain C ABI bridge → libponko_sd.so
#
# 此前这两个 .so 是手工编译后塞进 jniLibs/ 的，容易出现「改了 C++ 源码但 .so 没重编」
# 的静默错配（包能构建成功，跑起来却是旧行为）。改成这里现场编译后，
# 每次 assembleRelease 的产物必然与当前源码一致。

cmake_minimum_required(VERSION 3.22.1)
project(ponko_sd CXX C)

# ---------------------------------------------------------------------------
# 编译标志
# ---------------------------------------------------------------------------
# The Exynos 2400e profile keeps ARMv8.6-A + dotprod + I8MM + FP16 enabled for GGML's
# ARM quant kernels. Keep this ISA profile unchanged for the Exynos build.
# Cortex-X4, Cortex-A720 and Cortex-A520 implement I8MM, so this ISA profile is valid
# across the Exynos 2400e's heterogeneous CPU clusters. Disable the profile for generic
# ARM64 builds: -DPONKO_EXYNOS_2400E_CPU_PROFILE=OFF.
option(PONKO_EXYNOS_2400E_CPU_PROFILE
       "Tune native code for Exynos 2400e (Cortex-X4/A720/A520 with I8MM)" ON)
option(PONKO_ENABLE_VULKAN
       "Build the Vulkan backend and its shader toolchain" OFF)
option(PONKO_ENABLE_RUST_CPU_KERNELS
       "Link Ponko's Exynos 2400E Rust kernels into GGML CPU MUL_MAT" OFF)
set(PONKO_CPU_KERNELS_STATIC_LIB "" CACHE FILEPATH
    "Path to libponko_cpu_kernels.a built for Android arm64-v8a")
if (PONKO_ENABLE_RUST_CPU_KERNELS)
    if (NOT PONKO_EXYNOS_2400E_CPU_PROFILE)
        message(FATAL_ERROR "Ponko Rust CPU kernels currently target the Exynos 2400E profile only")
    endif()
    if (NOT EXISTS "${PONKO_CPU_KERNELS_STATIC_LIB}")
        message(FATAL_ERROR "PONKO_ENABLE_RUST_CPU_KERNELS requires PONKO_CPU_KERNELS_STATIC_LIB to exist")
    endif()
    # Define the opt-in hook before GGML's subdirectories are configured.
    add_compile_definitions(PONKO_ENABLE_RUST_CPU_KERNELS=1)
endif()
if (PONKO_EXYNOS_2400E_CPU_PROFILE)
    set(PONKO_CPU_ARCH_FLAGS -march=armv8.6-a+dotprod+i8mm+fp16 -mtune=cortex-x4)
    message(STATUS "Ponko CPU profile: Exynos 2400e (ARMv8.6-A + dotprod + I8MM + FP16, tuned for Cortex-X4)")
else()
    set(PONKO_CPU_ARCH_FLAGS -march=armv8.2-a+dotprod+fp16)
    message(STATUS "Ponko CPU profile: generic ARMv8.2-A (dotprod + FP16)")
endif()

# Keep GGML's own ARM feature detection aligned with the same Exynos ISA profile.
# This controls backend feature probes and architecture-specific kernel selection;
# global add_compile_options alone is not enough for every GGML subcomponent.
if (PONKO_EXYNOS_2400E_CPU_PROFILE)
    set(GGML_CPU_ARM_ARCH "armv8.6-a+dotprod+i8mm+fp16" CACHE STRING "" FORCE)
else()
    set(GGML_CPU_ARM_ARCH "" CACHE STRING "" FORCE)
endif()

# Android's toolchain can define CMAKE_*_FLAGS_DEBUG as normal variables, so a
# CACHE-only override does not necessarily change the flags used by Ninja. Apply
# Debug optimization directly to compile rules, while preserving the default -g.
add_compile_options(
    ${PONKO_CPU_ARCH_FLAGS}
    "$<$<CONFIG:Debug>:-O3>"
    "$<$<CONFIG:Debug>:-DNDEBUG>"
)

# Apply the compiler cache to the whole native project, not just GGML, when
# ccache/sccache is installed. Cache misses or absent tools fall back normally.
option(PONKO_ENABLE_COMPILER_CACHE "Use ccache/sccache for native compilation when available" ON)
if (PONKO_ENABLE_COMPILER_CACHE)
    if (NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
        find_program(PONKO_CCACHE_EXECUTABLE NAMES ccache)
        find_program(PONKO_SCCACHE_EXECUTABLE NAMES sccache)
        if (PONKO_CCACHE_EXECUTABLE)
            set(_ponko_cache_launcher "${PONKO_CCACHE_EXECUTABLE}")
        elseif (PONKO_SCCACHE_EXECUTABLE)
            set(_ponko_cache_launcher "${PONKO_SCCACHE_EXECUTABLE}")
        else ()
            set(_ponko_cache_launcher "")
        endif ()

        if (_ponko_cache_launcher)
            set(CMAKE_C_COMPILER_LAUNCHER "${_ponko_cache_launcher}")
            set(CMAKE_CXX_COMPILER_LAUNCHER "${_ponko_cache_launcher}")
            set(CMAKE_C_COMPILER_LAUNCHER "${_ponko_cache_launcher}" CACHE STRING "" FORCE)
            set(CMAKE_CXX_COMPILER_LAUNCHER "${_ponko_cache_launcher}" CACHE STRING "" FORCE)
            set(PONKO_DETECTED_COMPILER_CACHE "${_ponko_cache_launcher}" CACHE INTERNAL "Ponko-selected compiler cache" FORCE)
            message(STATUS "Ponko compiler cache: ${_ponko_cache_launcher}")
        else ()
            message(STATUS "Ponko compiler cache not found; native compilation will proceed uncached.")
        endif ()
    else ()
        message(STATUS "Ponko compiler cache: honoring existing CMake compiler launcher.")
    endif ()
elseif (DEFINED PONKO_DETECTED_COMPILER_CACHE)
    # Remove only the launcher previously selected by Ponko, never a user-supplied tool.
    if ("${CMAKE_C_COMPILER_LAUNCHER}" STREQUAL "${PONKO_DETECTED_COMPILER_CACHE}")
        unset(CMAKE_C_COMPILER_LAUNCHER CACHE)
        set(CMAKE_C_COMPILER_LAUNCHER "")
    endif ()
    if ("${CMAKE_CXX_COMPILER_LAUNCHER}" STREQUAL "${PONKO_DETECTED_COMPILER_CACHE}")
        unset(CMAKE_CXX_COMPILER_LAUNCHER CACHE)
        set(CMAKE_CXX_COMPILER_LAUNCHER "")
    endif ()
    unset(PONKO_DETECTED_COMPILER_CACHE CACHE)
    message(STATUS "Ponko compiler cache disabled.")
endif ()

# 强制各 Release 变体都是 -O3 且不带调试符号。
# （AGP 给 release 变体传的可能是 RelWithDebInfo = -O2 -g，性能与体积都不理想）
foreach(_cfg RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_C_FLAGS_${_cfg}   "-O3 -DNDEBUG" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_${_cfg} "-O3 -DNDEBUG" CACHE STRING "" FORCE)
endforeach()


# 限制并行度：源码含若干超大翻译单元，8GB 内存的机器上 -j4 会 OOM（LLVM ERROR: out of memory）
set_property(GLOBAL PROPERTY JOB_POOLS ponko_compile=2)
set(CMAKE_JOB_POOL_COMPILE ponko_compile)

# ---------------------------------------------------------------------------
# sd.cpp 选项（必须在 add_subdirectory 之前用 CACHE ... FORCE 设好）
# ---------------------------------------------------------------------------
set(SD_BUILD_EXAMPLES        OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS     ON  CACHE BOOL "" FORCE)  # 产出 libstable-diffusion.so
set(SD_BUILD_SHARED_GGML_LIB OFF CACHE BOOL "" FORCE)  # ggml 静态并入，少一个 .so
set(SD_USE_SYSTEM_GGML       OFF CACHE BOOL "" FORCE)
set(SD_WEBP   OFF CACHE BOOL "" FORCE)
set(SD_WEBM   OFF CACHE BOOL "" FORCE)
set(SD_CUDA   OFF CACHE BOOL "" FORCE)
set(SD_VULKAN ${PONKO_ENABLE_VULKAN} CACHE BOOL "" FORCE)  # OFF builds a CPU-only variant; ON retains Vulkan + shader-toolchain behavior.
set(SD_OPENCL OFF CACHE BOOL "" FORCE)
set(SD_METAL  OFF CACHE BOOL "" FORCE)
set(SD_SYCL   OFF CACHE BOOL "" FORCE)
set(SD_HIPBLAS OFF CACHE BOOL "" FORCE)
set(SD_MUSA   OFF CACHE BOOL "" FORCE)

set(GGML_OPENMP         ON  CACHE BOOL "" FORCE)  # OpenMP 多线程（运行时用 NDK 的 libomp.so）
set(GGML_CPU            ON  CACHE BOOL "" FORCE)
set(GGML_CPU_REPACK     ON  CACHE BOOL "" FORCE)
# Benchmark branch only: test Arm KleidiAI's I8MM/DOTPROD kernels on Exynos 2400e.
set(GGML_CPU_KLEIDIAI   ON  CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE         OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE         ${PONKO_ENABLE_COMPILER_CACHE} CACHE BOOL "" FORCE)  # Use ccache/sccache when installed; gracefully falls back when absent.

# ---------------------------------------------------------------------------
# Apply the pinned upstream source plus Ponko's feature overlay before configuration.
# The submodule commit itself is pinned by the repository gitlink; this marker is keyed to
# the exact overlay content so a changed patch is reapplied from a clean upstream worktree.
set(PONKO_SDCpp_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sd")
set(PONKO_SDCpp_PATCH "${CMAKE_CURRENT_SOURCE_DIR}/ponko-sdcpp-patches/upstream-port.patch")
set(PONKO_SDCpp_PATCH_MARKER "${PONKO_SDCpp_DIR}/.ponko-patches-applied")
file(SHA256 "${PONKO_SDCpp_PATCH}" PONKO_SDCpp_PATCH_SHA)
set(PONKO_SDCpp_EXPECTED_MARKER "upstream=77b72a2efed78f55de26892c7e909cee001e8b83;patch=${PONKO_SDCpp_PATCH_SHA}\n")
set(PONKO_SDCpp_PATCH_NEEDED ON)
if(EXISTS "${PONKO_SDCpp_PATCH_MARKER}")
    file(READ "${PONKO_SDCpp_PATCH_MARKER}" PONKO_SDCpp_APPLIED_MARKER)
    if(PONKO_SDCpp_APPLIED_MARKER STREQUAL PONKO_SDCpp_EXPECTED_MARKER)
        set(PONKO_SDCpp_PATCH_NEEDED OFF)
    else()
        # CMake may be reconfigured in the same checkout after the overlay changes.
        # Reset only the pinned submodule worktree, not any tracked Ponko app sources.
        find_program(PONKO_GIT_EXECUTABLE NAMES git)
        if(NOT PONKO_GIT_EXECUTABLE)
            message(FATAL_ERROR "Git is required to refresh the stable-diffusion.cpp feature overlay.")
        endif()
        execute_process(
            COMMAND "${PONKO_GIT_EXECUTABLE}" checkout -- .
            WORKING_DIRECTORY "${PONKO_SDCpp_DIR}"
            RESULT_VARIABLE PONKO_SDCpp_RESET_RESULT
        )
        if(NOT PONKO_SDCpp_RESET_RESULT EQUAL 0)
            message(FATAL_ERROR "Could not reset the pinned stable-diffusion.cpp worktree before reapplying the feature overlay.")
        endif()
        if(EXISTS "${PONKO_SDCpp_DIR}/ggml/CMakeLists.txt")
            execute_process(
                COMMAND "${PONKO_GIT_EXECUTABLE}" checkout -- .
                WORKING_DIRECTORY "${PONKO_SDCpp_DIR}/ggml"
                RESULT_VARIABLE PONKO_GGML_RESET_RESULT
            )
            if(NOT PONKO_GGML_RESET_RESULT EQUAL 0)
                message(FATAL_ERROR "Could not reset the pinned GGML worktree before reapplying the feature overlay.")
            endif()
        endif()
    endif()
endif()

if(PONKO_SDCpp_PATCH_NEEDED)
    find_program(PONKO_PATCH_EXECUTABLE NAMES patch)
    if(NOT PONKO_PATCH_EXECUTABLE)
        message(FATAL_ERROR "Ponko requires the patch utility to port native feature changes.")
    endif()
    # Git may check out upstream files with CRLF, while this overlay is LF.
    # Normalize the two whole-file replacement targets before invoking GNU patch.
    foreach(_ponko_patch_target
            "${PONKO_SDCpp_DIR}/src/convert.cpp")
        file(READ "${_ponko_patch_target}" _ponko_patch_source)
        string(REPLACE "\r\n" "\n" _ponko_patch_source "${_ponko_patch_source}")
        file(WRITE "${_ponko_patch_target}" "${_ponko_patch_source}")
    endforeach()
    execute_process(
        COMMAND "${PONKO_PATCH_EXECUTABLE}" --batch --forward -p6 -F3
                -i "${PONKO_SDCpp_PATCH}"
        WORKING_DIRECTORY "${PONKO_SDCpp_DIR}"
        RESULT_VARIABLE PONKO_SDCpp_PATCH_RESULT
        OUTPUT_VARIABLE PONKO_SDCpp_PATCH_STDOUT
        ERROR_VARIABLE PONKO_SDCpp_PATCH_STDERR
    )
    if(NOT PONKO_SDCpp_PATCH_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to apply the Ponko feature patch.\n${PONKO_SDCpp_PATCH_STDOUT}\n${PONKO_SDCpp_PATCH_STDERR}")
    endif()
    # Keep Ponko's TCD scheduler exposed without renumbering upstream scheduler values.
    # The overlay usually adds this already; the guarded fallback also makes patch revisions
    # that omit the enum hunk deterministic rather than duplicating the entry.
    set(PONKO_SD_HEADER "${PONKO_SDCpp_DIR}/include/stable-diffusion.h")
    file(READ "${PONKO_SD_HEADER}" PONKO_SD_HEADER_TEXT)
    string(FIND "${PONKO_SD_HEADER_TEXT}" "    TCD_SCHEDULER,\n    SCHEDULER_COUNT" PONKO_TCD_ENUM_PRESENT)
    if(PONKO_TCD_ENUM_PRESENT EQUAL -1)
        string(FIND "${PONKO_SD_HEADER_TEXT}" "    LLADA_IMAGE_SCHEDULER,\n    SCHEDULER_COUNT" PONKO_TCD_ENUM_ANCHOR)
        if(PONKO_TCD_ENUM_ANCHOR EQUAL -1)
            message(FATAL_ERROR "Could not locate the stable-diffusion.cpp scheduler enum for TCD compatibility.")
        endif()
        string(REPLACE "    LLADA_IMAGE_SCHEDULER,\n    SCHEDULER_COUNT"
                       "    LLADA_IMAGE_SCHEDULER,\n    TCD_SCHEDULER,\n    SCHEDULER_COUNT"
                       PONKO_SD_HEADER_TEXT "${PONKO_SD_HEADER_TEXT}")
        file(WRITE "${PONKO_SD_HEADER}" "${PONKO_SD_HEADER_TEXT}")
    endif()

    set(PONKO_SD_VERSION_CPP "${PONKO_SDCpp_DIR}/src/stable-diffusion.cpp")
    file(READ "${PONKO_SD_VERSION_CPP}" PONKO_SD_VERSION_TEXT)
    string(FIND "${PONKO_SD_VERSION_TEXT}" "    \"tcd\",\n};" PONKO_TCD_NAME_PRESENT)
    if(PONKO_TCD_NAME_PRESENT EQUAL -1)
        string(FIND "${PONKO_SD_VERSION_TEXT}" "    \"llada_image\",\n};" PONKO_TCD_NAME_ANCHOR)
        if(PONKO_TCD_NAME_ANCHOR EQUAL -1)
            message(FATAL_ERROR "Could not locate stable-diffusion.cpp scheduler names for TCD compatibility.")
        endif()
        string(REPLACE "    \"llada_image\",\n};"
                       "    \"llada_image\",\n    \"tcd\",\n};"
                       PONKO_SD_VERSION_TEXT "${PONKO_SD_VERSION_TEXT}")
        file(WRITE "${PONKO_SD_VERSION_CPP}" "${PONKO_SD_VERSION_TEXT}")
    endif()

    file(WRITE "${PONKO_SDCpp_PATCH_MARKER}" "${PONKO_SDCpp_EXPECTED_MARKER}")
    message(STATUS "Ponko native feature patches applied to upstream source")
else()
    message(STATUS "Ponko native feature patches already applied for this patch revision")
endif()

# 引入 sd.cpp
# ---------------------------------------------------------------------------
# AGP 期望 .so 落在 CMAKE_LIBRARY_OUTPUT_DIRECTORY；而 sd.cpp 顶层会把它改成
# <build>/bin，导致产物打不进 APK。所以先记下 AGP 给的值，之后再拨回去。
set(PONKO_SO_OUT "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "Ponko: AGP .so output dir = '${PONKO_SO_OUT}'")

# ---------------------------------------------------------------------------
# Vulkan 后端依赖自举（不要求安装 Vulkan SDK）
# ---------------------------------------------------------------------------
# ggml-vulkan 需要两样东西，NDK 里其实都自带了：
#   1) glslc：把 .comp 着色器编译成 SPIR-V —— NDK 的 shader-tools/<host>/glslc[.exe]
#   2) SPIRV-Headers：一个 CMake 包配置（find_package(SPIRV-Headers CONFIG)）
#      —— NDK 在 sources/third_party/shaderc 里带了 spirv-headers 头文件，这里就地生成最小配置
# 另外交叉编译时 ggml 还要一个 host 工具（vulkan-shaders-gen）：Windows 需要 cl.exe，
# Linux/macOS 用自带的 gcc/clang 即可。构建命令里带上 MSVC 环境（vcvars64）就行。
if (SD_VULKAN)
    # NDK 只带 Vulkan 的 C 头，没有 C++ 头（vulkan.hpp）。这里用仓库里自带的 Vulkan-Hpp
    # （与 NDK 的 vulkan_core.h 同为 1.3.275）。只追加 include 路径，不覆盖 NDK 的 C 头。
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/include")
    # ggml-vulkan 还会 include SPIRV-Headers 的 C++ 头（spirv/unified1/spirv.hpp），NDK 里自带有
    set(_ponko_spirv_hpp_inc "${ANDROID_NDK}/sources/third_party/shaderc/third_party/spirv-tools/external/spirv-headers/include")
    if (EXISTS "${_ponko_spirv_hpp_inc}/spirv/unified1/spirv.hpp")
        include_directories("${_ponko_spirv_hpp_inc}")
    else ()
        message(WARNING "Ponko: spirv.hpp not found in the NDK (${_ponko_spirv_hpp_inc})")
    endif ()

    # 注意：交叉编译时 WIN32 指的是【目标平台】（这里是 Android），所以必须用 host 判断
    if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
        # 先把 ExternalProject 里 define_property(... INHERITED) 的定义加载进来，
        # 否则 EP_BASE 会被当成普通属性设上、不会继承给子目录（实测无效）。
        include(ExternalProject)
        # Windows 的 MAX_PATH：ggml 的 host 工具（vulkan-shaders-gen）用 ExternalProject 构建，
        # 默认会落在 <build>/.cxx/<config>/<hash>/<abi>/sd/ggml/... 这种超深目录里，
        # MSVC 因对象文件路径超过 250 字符直接报 C1083。用 EP_BASE 把它挪到短目录（只影响这个 host 工具）。
        set(_ponko_ep_base "$ENV{TEMP}/ponko-ep")
        if (NOT _ponko_ep_base OR _ponko_ep_base STREQUAL "/ponko-ep")
            set(_ponko_ep_base "C:/ponko-ep")
        endif ()
        file(TO_CMAKE_PATH "${_ponko_ep_base}" _ponko_ep_base)
        set_property(DIRECTORY PROPERTY EP_BASE "${_ponko_ep_base}")
        message(STATUS "Ponko: external project base = ${_ponko_ep_base}")
    endif ()

    if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
        set(_ponko_host_tag "windows-x86_64")
        set(_ponko_glslc_name "glslc.exe")
    elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        set(_ponko_host_tag "darwin-x86_64")
        set(_ponko_glslc_name "glslc")
    else ()
        set(_ponko_host_tag "linux-x86_64")
        set(_ponko_glslc_name "glslc")
    endif ()

    # Optional host glslc override. NDK-bundled shaderc can be too old to compile
    # GL_KHR_cooperative_matrix. A newer host compiler only emits SPIR-V at build time;
    # the Samsung Vulkan driver still decides whether the optimized kernels can run.
    if (DEFINED ENV{PONKO_VULKAN_GLSLC} AND NOT "$ENV{PONKO_VULKAN_GLSLC}" STREQUAL "")
        set(_ponko_host_glslc "$ENV{PONKO_VULKAN_GLSLC}")
        if (EXISTS "${_ponko_host_glslc}")
            set(Vulkan_GLSLC_EXECUTABLE "${_ponko_host_glslc}" CACHE FILEPATH "Ponko host glslc override" FORCE)
            message(STATUS "Ponko: using host glslc override = ${Vulkan_GLSLC_EXECUTABLE}")
        else ()
            message(WARNING "Ponko: PONKO_VULKAN_GLSLC path does not exist (${_ponko_host_glslc}); falling back to the NDK glslc")
        endif ()
    endif ()

    if (NOT Vulkan_GLSLC_EXECUTABLE)
        set(_ponko_glslc "${ANDROID_NDK}/shader-tools/${_ponko_host_tag}/${_ponko_glslc_name}")
        if (EXISTS "${_ponko_glslc}")
            set(Vulkan_GLSLC_EXECUTABLE "${_ponko_glslc}" CACHE FILEPATH "glslc bundled with the NDK" FORCE)
            message(STATUS "Ponko: glslc = ${_ponko_glslc}")
        else ()
            message(WARNING "Ponko: glslc not found in the NDK (${_ponko_glslc}); set Vulkan_GLSLC_EXECUTABLE or install the Vulkan SDK")
        endif ()
    endif ()

    if (NOT SPIRV-Headers_DIR)
        set(_ponko_spirv_inc "${ANDROID_NDK}/sources/third_party/shaderc/third_party/spirv-tools/external/spirv-headers/include")
        if (EXISTS "${_ponko_spirv_inc}/spirv/unified1/spirv.hpp")
            set(_ponko_cfg_dir "${CMAKE_BINARY_DIR}/ponko-spirv-headers")
            file(MAKE_DIRECTORY "${_ponko_cfg_dir}")
            file(WRITE "${_ponko_cfg_dir}/SPIRV-HeadersConfig.cmake"
"# generated by Ponko: minimal SPIRV-Headers config (headers come from the NDK)
set(SPIRV-Headers_FOUND TRUE)
set(SPIRV-Headers_INCLUDE_DIR \"${_ponko_spirv_inc}\")
set(SPIRV-Headers_INCLUDE_DIRS \"${_ponko_spirv_inc}\")
if(NOT TARGET SPIRV-Headers::SPIRV-Headers)
    add_library(SPIRV-Headers::SPIRV-Headers INTERFACE IMPORTED)
    set_target_properties(SPIRV-Headers::SPIRV-Headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES \"${_ponko_spirv_inc}\")
endif()
")
            set(SPIRV-Headers_DIR "${_ponko_cfg_dir}" CACHE PATH "generated minimal SPIRV-Headers config" FORCE)
            message(STATUS "Ponko: SPIRV-Headers config generated at ${_ponko_cfg_dir}")
        else ()
            message(WARNING "Ponko: SPIRV headers not found in the NDK (${_ponko_spirv_inc}); set SPIRV-Headers_DIR manually")
        endif ()
    endif ()
endif ()

add_subdirectory(sd)

if (PONKO_ENABLE_RUST_CPU_KERNELS)
    add_library(ponko_cpu_kernels STATIC IMPORTED GLOBAL)
    set_target_properties(ponko_cpu_kernels PROPERTIES
        IMPORTED_LOCATION "${PONKO_CPU_KERNELS_STATIC_LIB}")
    if (NOT TARGET ggml-cpu)
        message(FATAL_ERROR "Ponko Rust CPU integration expected the GGML target ggml-cpu")
    endif()
    target_link_libraries(ggml-cpu PRIVATE ponko_cpu_kernels)
endif()

if(PONKO_SO_OUT)
    foreach(_sfx "" _RELEASE _RELWITHDEBINFO _DEBUG)
        set_target_properties(stable-diffusion PROPERTIES
            "LIBRARY_OUTPUT_DIRECTORY${_sfx}" "${PONKO_SO_OUT}")
    endforeach()
endif()

# ---------------------------------------------------------------------------
# C ABI bridge → libponko_sd.so
# ---------------------------------------------------------------------------
# Plain C ABI consumed by the Rust/UniFFI facade. The app-facing JNI entry points
# are no longer part of this target; Kotlin reaches this bridge through Rust.
add_library(ponko_sd SHARED ponko_sd_c_api.cpp)
target_include_directories(ponko_sd PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/sd/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/sd/ggml/include"
)
if (SD_VULKAN)
    target_compile_definitions(ponko_sd PRIVATE PONKO_HAS_VULKAN=1)
endif()
target_link_libraries(ponko_sd PRIVATE stable-diffusion android log)
