# Toolchain — verified on this machine (Windows 11, git-bash, no admin)

Every path and version below was checked on 2026-11-15. Host is `C:\Users\harri`; the shell your
`terminal` tool gets is **git-bash (MSYS)**, so POSIX syntax works but **native Windows binaries
do not get `/c/...` path translation** — pass `C:/...` paths to `cmake.exe`, `node`, `python`,
`git` arguments, and use `pwd -W` to build them.

## Android

| tool | path | version |
|---|---|---|
| adb | `C:/Users/harri/AppData/Local/Android/Sdk/platform-tools/adb.exe` | 1.0.41 (37.0.1) |
| NDK | `C:/Users/harri/AppData/Local/Android/Sdk/ndk/27.3.13750724` | r27d |
| CMake + ninja | `C:/Users/harri/AppData/Local/Android/Sdk/cmake/3.22.1/bin/{cmake,ninja}.exe` | 3.22.1 |
| SDK platforms | `android-34`, build-tools `34.0.0` | — |
| JDK (for sdkmanager/gradle only) | `C:/Users/harri/tools/jdk-17.0.21+9` | Temurin 17.0.21 |

`adb` is **not** on `PATH`; `tools/probe/probe_host.py` resolves it itself. If you need it in a
shell: `export PATH="$PATH:/c/Users/harri/AppData/Local/Android/Sdk/platform-tools"`.

`sdkmanager` needs `JAVA_HOME`, else it dies with `UnsupportedClassVersionError` (system java is 8):

```
export JAVA_HOME="C:/Users/harri/tools/jdk-17.0.21+9"
yes | "$SDK/cmdline-tools/latest/bin/sdkmanager.bat" --install "ndk;27.3.13750724"
```

NDK headers that matter (all present, verified):
`sysroot/usr/include/android/{neuralnetworks.h,NeuralNetworksTypes.h,hardware_buffer.h}`,
`sysroot/usr/lib/aarch64-linux-android/34/libneuralnetworks.so`.
There is **no** `libOpenCL.so` stub in the NDK — `dlopen("libOpenCL.so")` at runtime.

### Building a native executable (no Gradle, no APK) — the pattern used here

```
tools/android/probe/build.sh        # cmake -G Ninja with the NDK toolchain file
adb push tools/android/probe/build/probe /data/local/tmp/hv_probe
adb shell chmod 755 /data/local/tmp/hv_probe
adb shell /data/local/tmp/hv_probe all
```
This is how all device-side measurement code should be built and run: plain PIE executables in
`/data/local/tmp`. Fast to iterate, no app signing, and OpenCL/NNAPI/`AHardwareBuffer` all work
from a shell-launched binary (confirm on first device attach — if a driver refuses, the fallback
is an APK, which costs a Gradle+JDK setup).

## Python for host-side work (Phase 1, Phase 4)

Three interpreters exist. Pick deliberately:

| interpreter | has | use for |
|---|---|---|
| `python` = 3.11.16 (`AppData/Local/Programs/Python/Python311`) | numpy 2.4.3, huggingface_hub 1.29, safetensors **not** installed | HF downloads, quick scripts (used by the probe + inspector) |
| `python3` = 3.14.3 | numpy | nothing yet |
| `C:/Users/harri/ComfyUI-Zluda/venv/Scripts/python.exe` | **torch 2.7.0+cu118 (ZLUDA → AMD RX 6700 XT works)**, diffusers 0.42.0, transformers 5.15.1, safetensors 0.8.0, accelerate 1.15.0, scikit-image 0.26 | quantisation, FP32 reference inference, PSNR/SSIM |

`diffusers 0.42.0` has `WanTransformer3DModel`, `AutoencoderKLWan`, `WanPipeline`,
`FlowMatchEulerDiscreteScheduler` (verified) — so the FP32 reference path needs **no** extra
install. `lpips` is **missing** → `pip install lpips` into the ComfyUI venv (needs torch, already
there). Machine RAM 32 GB, free disk ~199 GB before the 30 GB of weights.

Do not create a new venv for Phase 1 unless you must: the ComfyUI venv is the only one with torch,
and model work is dominated by I/O anyway.

## Weights

```
weights_raw/DCM_WAN/{transformer,vae,text_encoder,tokenizer,scheduler,model_index.json}
```
from HF **`cszy98/DCM`** (public, no token). ~6.7 GB for transformer+VAE, ~22.7 GB for the umT5-xxl
text encoder (needed only to precompute prompt embeddings — the paper does exactly that offline and
does not keep the encoder resident). Download command used:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="cszy98/DCM", local_dir="weights_raw",
                  allow_patterns=["DCM_WAN/transformer/*", "DCM_WAN/vae/*",
                                  "DCM_WAN/text_encoder/*", "DCM_WAN/tokenizer/*",
                                  "DCM_WAN/scheduler/*", "DCM_WAN/model_index.json"])
```
`DCM_HY/*` (26 GB transformer) is the Hunyuan variant — **not** this project.

Verified facts about the payload: transformer is **FP32** (5.73 GiB, 1,539,041,408 params),
VAE is `AutoencoderKLWan` (16-ch latents, 8× spatial / 4× temporal).

## Reference environments

- `third_party/DCM` — model + scheduler + the 6-step inference script (`scripts/inference/inference_wan.sh`,
  args in `docs/02_model_spec.md`). Install is `pip install -e .` + `flash-attn`; **flash-attn will
  not build on ZLUDA** — for host reference work import `fastvideo.models.wan_hf` directly and use
  PyTorch SDPA; only the training path needs flash-attn.
- `third_party/Wan2.1` — official Wan VAE / decoder reference (3-D causal conv + cache semantics).
- `third_party/ncnn` — Mali OpenCL/Vulkan INT8 kernels (`src/layer/arm/*dot*`, `gemm`); the closest
  thing to a working "Mali INT8 GEMM" reference.
- `third_party/llama.cpp` (sparse: `ggml/src/ggml-opencl`) — OpenCL kernel patterns + Android NDK
  CMake layout; `ggml-opencl.cpp` has `cl_arm_integer_dot_product` usage and a dp4a path.
- `third_party/htp-ops-lib` — Hexagon HMX/HVX reference. **Not portable to Pixel**; read it for
  tiling/attention structure only.
- `third_party/ndk-samples` — `nn-samples` has the NNAPI + `AHardwareBuffer` examples that compile
  against this NDK.
- `third_party/opencl-headers` — Khronos headers if you want real `CL/cl.h` instead of hand-declared
  types (the probe hand-declares on purpose so it links with nothing).

## Getting-started command sequence

```bash
cd C:/Users/harri/Documents/GitHub/HeteroVDiff_Pixel_6a
python phase1_quant/inspect_weights.py --manifest configs/block_manifest.json
python tools/probe/probe_host.py --preflight          # is the phone attached?
python tools/probe/probe_host.py --build              # build + run the full Gate-0 probe
python tools/probe/probe_host.py --cooldown 30        # thermal gate before any timed run
```
