# Reference repositories (vendored in `third_party/`, gitignored)

Re-clone with `docs/07_repos.sh`. Commit hashes are what is on disk now — pin them in the report if
you cite code from them.

| repo | commit | what it is | how to use it here | what **not** to expect |
|---|---|---|---|---|
| `Vchitect/DCM` | `c7039e8` | official DCM code: model, dual-expert training/distillation, 6-step sampler | the model definition (`fastvideo/models/wan_hf/modeling_wan.py`), the 6-step scheduler (`fastvideo/distill/solver.py` → `InferencePCMFMScheduler`), the exact inference recipe (`scripts/inference/inference_wan.sh`), expert-switch rule (`inference_wan.py: t<=978 → set_adapter('lora1')`) | it needs `flash-attn`, which will not build on ZLUDA; import the model directly and use SDPA |
| `Wan-Video/Wan2.1` | `9737cba` | upstream Wan2.1: the VAE/decoder, text encoder preprocessing | ground truth for `AutoencoderKLWan` causal-3D-conv + temporal cache semantics, and the prompt-embedding preprocess | the 1.3B weights there are *not* DCM-distilled; do not swap them in |
| `cszy98/DCM` (HF, not git) | — | the weights: `DCM_WAN/{transformer,vae,text_encoder,tokenizer}` | the actual payload (`weights_raw/`) | 22.7 GB text encoder is only needed to make prompt embeddings; the paper precomputes them offline |
| `Qualcomm`/`haozixu/htp-ops-lib` | `85eb88e` | Hexagon HMX/HVX op library the paper extends (its NPU kernels are "added to htp-ops-lib") | read the attention/HMX tiling structure and the two library bugs the paper had to fix (softmax exp LUT accuracy, FP32→FP16 cast interleave) | **nothing here runs on a Pixel** — no Hexagon, no FastRPC |
| `ggml-org/llama.cpp` (sparse: `ggml/src/ggml-opencl`) | `75fd7f9` | OpenCL kernels incl. `cl_arm_integer_dot_product` and dp4a paths; Android NDK CMake structure | kernel structure for INT8 dot products on Mali, `clBuildProgram` option handling, work-group sizing conventions | its kernels are tuned for GEMV/small batches; a Wan GEMM wants tiled operands |
| `Tencent/ncnn` | `d1f6ac6` | best-in-class mobile GPU (Vulkan/OpenCL) INT8 inference | the practical reference for **Mali** INT8 GEMM/conv tiling, `arm dot` intrinsics, packing layouts | Vulkan-first; the OpenCL path is secondary |
| `android/ndk-samples` | `abb0d8e` | official NDK samples | `nn-samples/` for NNAPI + `AHardwareBuffer` usage that actually compiles against NDK r27; `hello-jni`/`hello-libs` for CMake layout | the NNAPI samples are pre-deprecation; API calls still link |
| `KhronosGroup/OpenCL-Headers` | `82f6031` | Khronos `CL/cl.h` | include if you want real CL types (the probe hand-declares them so it links against nothing) | — |
| `Vchitect/VBench` | `fd18b3d` | the paper's *quality* benchmark (VBench + VBench-2.0 scorers) | optional: the paper reports VBench quality score. **Not required** for the course deliverable (PSNR/SSIM/LPIPS is) | heavy (torch, decord, detector/VLM checkpoints for VBench-2.0); budget a day |

Not vendored, worth pulling only if needed:

- `google-ai-edge/LiteRT` — the NNAPI successor; its issue tracker (#969, #6501) is the evidence
  that no public Tensor-TPU delegate exists.
- `ARM-software/ComputeLibrary` — `CL` kernels with `arm_dot` for GEMM/conv on Mali, if ncnn is not
  enough.
