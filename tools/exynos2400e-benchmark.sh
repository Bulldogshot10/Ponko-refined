#!/usr/bin/env sh
set -eu

OUT="${1-}"
if [ -z "$OUT" ]; then OUT="ponko-exynos-logcat.txt"; fi
RAW="$OUT.raw"
ADB="adb"

if ! command -v "$ADB" >/dev/null 2>&1; then
  echo "adb not found. Install Android platform-tools on this host." >&2
  exit 1
fi

APP_PID="$("$ADB" shell pidof -s com.litertchat.app 2>/dev/null | tr -d '\r')"
if [ -z "$APP_PID" ]; then
  echo "Ponko-refined must be open before capture so its process ID can be found." >&2
  exit 1
fi

echo "Clearing logcat buffer..."
"$ADB" logcat -c
echo "Capturing app PID $APP_PID to $RAW. Generate your fixed-settings test cases, then press Ctrl-C."
"$ADB" logcat --pid "$APP_PID" -v threadtime > "$RAW" &
LOGCAT_PID=$!
cleanup() {
  kill "$LOGCAT_PID" 2>/dev/null || true
  wait "$LOGCAT_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT
wait "$LOGCAT_PID" || true
grep -E 'BENCHMARK|PonkoSd|Using Conv2d direct in the vae model|Using standard VAE convolution path' "$RAW" > "$OUT" || true
echo "Parsing $OUT..."
python3 "$(dirname "$0")/exynos2400e-benchmark.py" "$OUT"
