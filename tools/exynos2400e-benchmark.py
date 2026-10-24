#!/usr/bin/env python3
"""Summarize Ponko Exynos benchmark logs into CSV and per-setting median reports."""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GEN_RE = re.compile(
    r"BENCHMARK generate_image elapsed_ms=(?P<elapsed>\d+)"
    r" mode=(?P<mode>\S+) size=(?P<size>\d+x\d+) steps=(?P<steps>\d+)"
    r" backend_request=(?P<backend>\S+) threads=(?P<threads>\d+)"
    r" vae_conv_direct=(?P<direct>[01]) sampler=(?P<sampler>-?\d+)"
    r" scheduler=(?P<scheduler>-?\d+) cfg=(?P<cfg>[0-9.]+)"
    r" lora_count=(?P<loras>\d+) seed=(?P<seed>-?\d+)"
    r" process_cpu_ms=(?P<cpu_ms>-?\d+) ok=(?P<ok>[01])"
)
VAE_RE = re.compile(
    r"BENCHMARK vae_stage stage=(?P<stage>\S+) elapsed_ms=(?P<elapsed>\d+)"
    r" threads=(?P<threads>\d+) tiled=(?P<tiled>[01])"
    r" input=(?P<input>\S+) output=(?P<output>\S+)"
)
DEVICE_RE = re.compile(r"BENCHMARK device stage=(?P<stage>\S+) (?P<metrics>.*)")
AFFINITY_RE = re.compile(r"BENCHMARK affinity status=(?P<status>.*)")

def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="captured adb logcat text")
    parser.add_argument("--csv", type=Path, default=Path("ponko-exynos-generations.csv"))
    parser.add_argument("--vae-csv", type=Path, default=Path("ponko-exynos-vae-stages.csv"))
    args = parser.parse_args()

    generations: list[dict[str, str]] = []
    vae_rows: list[dict[str, str]] = []
    device_rows: list[dict[str, str]] = []
    affinity_rows: list[str] = []
    direct_mode = "unknown"

    for line in read_lines(args.log):
        if "Using Conv2d direct in the vae model" in line:
            direct_mode = "direct"
        elif "Using standard VAE convolution path (direct Conv2D disabled)" in line:
            direct_mode = "standard"

        match = GEN_RE.search(line)
        if match:
            generations.append(match.groupdict())
            continue
        match = VAE_RE.search(line)
        if match:
            row = match.groupdict()
            row["vae_conv_mode"] = direct_mode
            vae_rows.append(row)
            continue
        match = DEVICE_RE.search(line)
        if match:
            device_rows.append(match.groupdict())
            continue
        match = AFFINITY_RE.search(line)
        if match:
            affinity_rows.append(match.group("status"))

    if generations:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(generations[0]))
            writer.writeheader()
            writer.writerows(generations)
    if vae_rows:
        with args.vae_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(vae_rows[0]))
            writer.writeheader()
            writer.writerows(vae_rows)

    print(f"generations={len(generations)} vae_stages={len(vae_rows)} device_snapshots={len(device_rows)}")
    if affinity_rows:
        print("recent affinity status:")
        for status in affinity_rows[-3:]:
            print(f"  {status}")

    keys = ("mode", "size", "steps", "backend", "threads", "direct", "sampler", "scheduler", "cfg", "loras")
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    failed = 0
    for row in generations:
        if row["ok"] != "1":
            failed += 1
            continue
        groups[tuple(row[k] for k in keys)].append(int(row["elapsed"]))

    if not groups:
        print("No successful generation benchmark records found. Check the APK's native build and captured log.", file=sys.stderr)
        return 2

    print("\nGeneration latency, grouped by exact settings:")
    for key, values in sorted(groups.items()):
        print(
            f"mode={key[0]} size={key[1]} steps={key[2]} backend={key[3]} threads={key[4]} "
            f"direct={key[5]} sampler={key[6]} scheduler={key[7]} cfg={key[8]} loras={key[9]} "
            f"n={len(values)} median_ms={statistics.median(values):.1f} "
            f"min_ms={min(values)} max_ms={max(values)}"
        )
    if failed:
        print(f"failed_generations={failed}")

    if device_rows:
        print("\nLast device/process snapshots:")
        for row in device_rows[-4:]:
            print(f"  {row['stage']}: {row['metrics']}")

    print(f"\nGeneration CSV: {args.csv}")
    if vae_rows:
        print(f"VAE-stage CSV: {args.vae_csv}")
        for mode in sorted({r["vae_conv_mode"] for r in vae_rows}):
            for stage in sorted({r["stage"] for r in vae_rows if r["vae_conv_mode"] == mode}):
                vals = [int(r["elapsed"]) for r in vae_rows if r["vae_conv_mode"] == mode and r["stage"] == stage]
                print(f"VAE mode={mode} stage={stage}: n={len(vals)} median_ms={statistics.median(vals):.1f} min_ms={min(vals)} max_ms={max(vals)}")
    print("\nCompare only matching model/settings. Alternate order and let the phone cool between test batches.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
