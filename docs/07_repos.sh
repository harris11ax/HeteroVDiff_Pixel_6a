#!/usr/bin/env bash
# Re-clone the reference repositories referenced by the blueprint + this project.
# Everything lands in third_party/ (gitignored). Safe to re-run.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p third_party && cd third_party || exit 1

clone() {  # url dir [extra args...]
  local url="$1" dir="$2"; shift 2
  if [ -d "$dir/.git" ]; then echo "skip  $dir"; return; fi
  echo "clone $dir"
  git clone --quiet --depth 1 "$@" "$url" "$dir" || echo "FAILED $dir"
}

# blueprint §3
clone https://github.com/Vchitect/DCM.git          DCM
clone https://github.com/haozixu/htp-ops-lib.git   htp-ops-lib      # blueprint calls it QualComm/htp-ops-lib
clone https://github.com/android/ndk-samples.git   ndk-samples
clone https://github.com/KhronosGroup/OpenCL-Headers.git opencl-headers

# llama.cpp: only the OpenCL backend (full clone is ~2 GB)
if [ ! -d llama.cpp/.git ]; then
  git clone --quiet --depth 1 --filter=blob:none --sparse https://github.com/ggerganov/llama.cpp.git llama.cpp
  (cd llama.cpp && git sparse-checkout set ggml/src/ggml-opencl)
fi

# additions that matter for the Pixel 6a target
clone https://github.com/Wan-Video/Wan2.1.git      Wan2.1           # VAE / decoder ground truth
clone https://github.com/Tencent/ncnn.git          ncnn             # Mali INT8 kernel reference
clone https://github.com/Vchitect/VBench.git       VBench           # optional: paper's quality suite

echo
echo "third_party contents:"
ls -1
