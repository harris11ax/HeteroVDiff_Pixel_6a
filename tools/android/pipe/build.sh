#!/usr/bin/env bash
# Build the on-device chunk-handoff selftest (P3.1), arm64-v8a / Android 34.
# Plain PIE executable, no Gradle/APK — same pattern as tools/android/npublock.
set -euo pipefail

SDK="${ANDROID_SDK_ROOT:-C:/Users/harri/AppData/Local/Android/Sdk}"
NDK="${ANDROID_NDK:-$SDK/ndk/27.3.13750724}"
CMAKE="$SDK/cmake/3.22.1/bin/cmake.exe"
NINJA="$SDK/cmake/3.22.1/bin/ninja.exe"
# cmake.exe/ninja.exe are native Windows binaries; MSYS does NOT translate
# /c/... arguments for them -> use pwd -W (C:/... form).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W)"

"$CMAKE" -S "$HERE" -B "$HERE/build" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$NINJA" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-34 \
  -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release

"$CMAKE" --build "$HERE/build"

echo "built: $HERE/build/handoff_selftest"
