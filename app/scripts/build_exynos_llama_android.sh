#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="5.1.0"
ARM_ARCH="armv8.6-a+dotprod+i8mm+fp16"
SDK_ROOT="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
NDK_ROOT="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}"
if [[ -z "$NDK_ROOT" && -n "$SDK_ROOT" ]]; then
  NDK_ROOT="$SDK_ROOT/ndk/27.3.13750724"
fi
if [[ ! -x "$NDK_ROOT/build/cmake/android.toolchain.cmake" ]]; then
  echo "Android NDK 27.3.13750724 not found at: $NDK_ROOT" >&2
  exit 1
fi

WORK="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/ponko-java-llama-${VERSION}-exynos"
BASE_AAR="$WORK/llama-android-${VERSION}.aar"
AAR_STAGE="$WORK/aar"
AAR_OUT="$ROOT/app/libs/llama-android-exynos.aar"
mkdir -p "$(dirname "$AAR_OUT")"
rm -rf "$WORK"
mkdir -p "$WORK"

echo "Cloning java-llama.cpp v${VERSION} (the version used by Ponko)."
git clone --depth 1 --branch "v${VERSION}" https://github.com/bernardladenthin/java-llama.cpp.git "$WORK/src"

echo "Building the chat JNI library for Exynos 2400e: ${ARM_ARCH}"
(
  cd "$WORK/src"
  chmod +x .github/build.sh
  export BUILD_JOBS="${PONKO_LLAMA_BUILD_JOBS:-2}"
  export USE_CACHE=false
  CMAKE_ARGS="-DOS_NAME=Linux-Android -DOS_ARCH=aarch64 -DCMAKE_TOOLCHAIN_FILE=${NDK_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU=ON -DGGML_CPU_ARM_ARCH=${ARM_ARCH} -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DGGML_CCACHE=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF"
  .github/build.sh "$CMAKE_ARGS"
)

LIB="$(find "$WORK/src/llama/src/main/resources" -type f -path '*/Linux-Android/aarch64/libjllama.so' -print -quit)"
if [[ -z "$LIB" || ! -s "$LIB" ]]; then
  echo "Exynos chat JNI library was not produced under the expected Linux-Android/aarch64 resource path." >&2
  find "$WORK/src/llama/src/main/resources" -type f -name 'libjllama.so' -print >&2 || true
  exit 1
fi
grep -Fq "GGML_CPU_ARM_ARCH:STRING=${ARM_ARCH}" "$WORK/src/llama/build/CMakeCache.txt"

OBJDUMP="$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objdump"
if [[ ! -x "$OBJDUMP" ]]; then
  OBJDUMP="$(find "$NDK_ROOT/toolchains/llvm/prebuilt" -type f -name llvm-objdump -print -quit)"
fi
if [[ -z "$OBJDUMP" || ! -x "$OBJDUMP" ]]; then
  echo "llvm-objdump was not found in the Android NDK." >&2
  exit 1
fi
"$OBJDUMP" -d "$LIB" > "$WORK/libjllama.disasm"
grep -Eq '[[:space:]]sdot[[:space:]]' "$WORK/libjllama.disasm" || {
  echo "DOTPROD opcodes were not found in libjllama.so; refusing to package a generic build." >&2
  exit 1
}
grep -Eq '[[:space:]]smmla[[:space:]]' "$WORK/libjllama.disasm" || {
  echo "I8MM SMMLA opcodes were not found in libjllama.so; refusing to package a generic build." >&2
  exit 1
}

echo "Downloading the matching v${VERSION} Android AAR as the Java-class/resource base."
curl --fail --location --retry 3 --retry-delay 2 \
  "https://repo.maven.apache.org/maven2/net/ladenthin/llama-android/${VERSION}/llama-android-${VERSION}.aar" \
  --output "$BASE_AAR"
mkdir -p "$AAR_STAGE"
unzip -q "$BASE_AAR" -d "$AAR_STAGE"
test -f "$AAR_STAGE/classes.jar"
test -f "$AAR_STAGE/jni/arm64-v8a/libjllama.so"
cp "$LIB" "$AAR_STAGE/jni/arm64-v8a/libjllama.so"
rm -f "$AAR_OUT"
(
  cd "$AAR_STAGE"
  zip -qr "$AAR_OUT" .
)
unzip -l "$AAR_OUT" | grep -F 'jni/arm64-v8a/libjllama.so'

echo "Built Exynos-targeted chat AAR: $AAR_OUT"
sha256sum "$LIB" "$AAR_OUT"
