#!/usr/bin/env python3
"""Source-level contract checks for the TAESD and CPU-thread controls."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}", file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS: {message}")


draw = read("app/src/main/java/com/litertchat/app/draw/DrawPage.kt")
engine = read("app/src/main/java/com/litertchat/app/draw/SdCppEngine.kt")
activity = read("app/src/main/java/com/litertchat/app/MainActivity.kt")
jni = read("app/src/main/cpp/sdcpp_jni.cpp")
header = read("app/src/main/cpp/sd/include/stable-diffusion.h")
diffusion = read("app/src/main/cpp/sd/src/pipeline/diffusion_engine.cpp")
builders = read("app/src/main/cpp/sd/src/pipeline/model_builders.cpp")

# The slot order is an ABI contract between Kotlin and C++.
keys_match = re.search(r"val SLOT_KEYS = arrayOf\((.*?)\)", engine, re.S)
require(keys_match is not None, "Kotlin SLOT_KEYS declaration exists")
kotlin_keys = re.findall(r'"([a-z0-9_]+)"', keys_match.group(1))

enum_match = re.search(r"enum SlotIndex\s*\{(.*?)\};", jni, re.S)
require(enum_match is not None, "JNI SlotIndex declaration exists")
native_slots = []
for name in re.findall(r"\b(SLOT_[A-Z0-9_]+)\b", enum_match.group(1)):
    if name == "SLOT_COUNT":
        continue
    native_slots.append(name.removeprefix("SLOT_").lower())

require(kotlin_keys == native_slots and len(kotlin_keys) == 13,
        f"Kotlin/native slot order matches all {len(kotlin_keys)} entries")

require('"taesd" -> R.string.draw_taesd_label' in draw,
        "TAESD slot has a UI label")
require('kind == KIND_SINGLE && family == FAM_SD15 -> listOf("model" to true, "vae" to false, "taesd" to false)' in draw,
        "TAESD is optional for single-file SD1.5 models")
require('p.taesd_path             = slotPtr[SLOT_TAESD];' in jni,
        "JNI forwards the selected TAESD file to sd_ctx_params_t")
require("const char* taesd_path;" in header,
        "bundled native API exposes taesd_path")
require('if (strlen(SAFE_STR(sd_ctx_params->taesd_path)) > 0)' in diffusion,
        "native model loader checks the TAESD path")
require('use_tae = true;' in diffusion,
        "successful TAESD load activates the TAE pipeline")
require('if (options.use_tae && !options.tae_preview_only)' in builders,
        "TAE can replace the final VAE decoder, not only preview")
tae_branch = builders.index('} else if (options.use_tae && !options.tae_preview_only) {')
standard_vae_branch = builders.index('LOG_INFO("using VAE for encoding / decoding")', tae_branch)
require(tae_branch < standard_vae_branch,
        "TAE opt-in branch is exclusive and selected before the standard VAE branch")

# CPU selection is persisted, bounded, and carried all the way to sd.cpp.
require('private const val KEY_CPU_THREADS = "drawCpuThreads"' in draw,
        "CPU-thread preference has a stable key")
require('prefs().getInt(KEY_CPU_THREADS, 4)' in draw,
        "CPU-thread selection is restored from preferences")
require('nThreads = value.coerceIn(1, maxThreads)' in draw,
        "CPU-thread selection is bounded")
require('putInt(KEY_CPU_THREADS, nThreads)' in draw,
        "CPU-thread selection is persisted")
require('drawPage?.setCpuThreads(threadOptions[position].toInt())' in activity,
        "thread selector writes the chosen value")
require('nThreads = nThreads,' in draw,
        "model load passes selected thread count into SdCppEngine.create")
require('nativeCreate(modelPath, vaePath, nThreads, wtype, flashAttn, backend, slots)' in engine,
        "Kotlin wrapper forwards the selected thread count to JNI")
require('p.n_threads    = (nThreads > 0) ? nThreads : sd_get_num_physical_cores();' in jni,
        "JNI forwards the requested count to sd.cpp")
require('int n_threads;' in header,
        "native context has an explicit thread-count parameter")

print("All TAESD and CPU-thread wiring checks passed.")
