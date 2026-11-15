# Phase plans, deliverables and verification gates

Rules for every phase: (a) it ends with a runnable artifact, (b) every claim in a report is backed
by a file in the repo, (c) nothing is called "done" until the gate command has been run and its
output committed under `results/`.

---

## Phase 0 — Preparation (mostly done; blocked on the phone)

| task | status |
|---|---|
| SDK/NDK/CMake/adb/JDK installed | ✅ done |
| Weights downloaded (`weights_raw/DCM_WAN`) | ✅ done |
| Reference repos cloned | ✅ done |
| Checkpoint inventory + manifest | ✅ done (`configs/block_manifest.json`) |
| Device capability probe built | ✅ done (`tools/android/probe/build/probe`) |
| **Pixel 6a attached, Gate-0 probe run** | ❌ **blocked** — needs USB debugging + `adb devices` shows the phone |

**Gate 0 command:** `python tools/probe/probe_host.py --build`
**Gate 0 output:** `docs/device/probe_<ts>.{json,md}` containing, at minimum:
`CL_DEV[0].HAS_ARM_DOT_INT8`, `CL_DEV[0].MAX_MEM_ALLOC_MiB`, `AHB_MAX_OK_MiB`,
`ZEROCOPY_NNAPI_MEMORY_rc`, `ZEROCOPY_CL_IMPORT_dev0_mem`, `NNAPI_DEV[*].NAME`,
`NNAPI_FC[32x1536x8960].ms_per_call` per device.

**Decision gate 0** (record the answer in `configs/design_locks.json`):
- zero-copy route: `A` (AHB + FDs) if `ZEROCOPY_CL_IMPORT_dev0_mem=OK` and NNAPI memory rc=0,
  else `B` (file mmap), else `C` (private copies — flag as Gate-2 failure)
- NPU stage exists: `NNAPI_DEV` list contains a non-`cpu` device that compiles an INT8 FC and is
  faster than the default path; otherwise Process B runs on CPU/XNNPACK and this is stated plainly

---

## Phase 1 — INT8 quantisation, block slicing, artifact export (host only, no phone needed)

Deliverables in `phase1_quant/`:

1. `quantize_int8.py` — for every linear weight `W ∈ R^{N_out×K}`:
   `s_j = max_i |W_ij| / 127` per **output column** `j`; `Q = clip(round(W/s_j), -127, 127)` (int8);
   write per-block region `artifacts/blocks/block_{l:02d}.bin` = concatenation of that block's INT8
   matrices in a documented order + `block_{l:02d}.scales.f32`; emit `manifest.json` with the exact
   tensor offsets inside each region. Keep bias/norms/AdaLN/patch-embed in **FP16** (they are 0.02 %
   of the parameters and quantising them costs quality for no memory).
2. `expert_factors.py` — extract rank-128 LoRA `A [128,1536]` / `B [1536,128]` per projection for
   all 30 blocks, plus the detail expert's `condition_embedder_lora` / `norm_out_lora` /
   `proj_out_lora` blocks; store FP16 in `artifacts/experts/detail/`. Emit the routing table
   `steps → expert` (`[semantic×3, detail×3]`).
3. `vae_export.py` — flatten `AutoencoderKLWan` into a layer list (causal Conv3d with kernel,
   stride, padding, temporal cache semantics, resnet blocks, attention, upsamplers) as JSON, plus
   the two decompositions: (a) *NPU plan* — each `Conv3d(k=3)` → 3 shifted `Conv2d` taps + adds;
   (b) *Mali plan* — direct 3D conv in OpenCL. Also emit the `latents_mean`/`latents_std` scaling
   and the 60×40-latent-pixel tile schedule.
4. `calibrate.py` — with the FP32 reference path (Phase 4.1) run one prompt and record, per block
   and per INT8 GEMM input, `max|x|/127` → `artifacts/act_scales.json`. Timestep-aware: run all 6
   steps, store per (step, block, gemm).

**Gate 1 (artifact integrity), all scripted in `phase1_quant/verify_artifact.py`:**
- total INT8 bytes = 1.296 GiB ± 1 MiB (measured reference: base 1,391,984,640 params);
- each region ≤ 47.29 MiB and ≤ `AHB_MAX_OK_MiB` from Gate 0;
- dequant(quant(W)) vs W: report per-GEMM relative Frobenius error and cosine similarity
  (expect cosine > 0.999, rel. error ≈ 3–5 % for per-column INT8 on Wan weights);
- an end-to-end FP32-vs-INT8-simulated forward on one 9-frame 256×256 prompt (weights quantised,
  activations FP16 like the GPU path) reaching **PSNR ≥ 28 dB / SSIM ≥ 0.92** on the decoded frames
  before any device work. If this fails on the host, no amount of device engineering will save it.

---

## Phase 2 — Rates, buffer ceilings, thermal protocol (needs the phone)

Deliverables in `phase2_profile/` + `tools/`:

1. `tools/probe/probe_host.py` — capability probe (done).
2. `phase2_profile/rate_probe.cpp` (device, built like the probe): times `T_G` (one DiT block on
   Mali, INT8, at the chosen `M`), `T_N` (one NNAPI block-range graph), and `T_V` (VAE decode of a
   9-frame latent through the chosen decoder path), **each alone and while the other engine is
   busy** — the paper's whole point is that co-run ≠ solo (NPU +10…19 %, GPU −1…2 %).
3. `phase2_profile/thermal.py` — cooldown polling + run bookkeeping (`--cooldown`, logs zone temps).
4. `phase2_profile/solve_cut.py` — **implemented** and self-tested: implements Eq. 4/5, evaluates
   every integer `k_b ∈ [1,179]`, reports the step-level and no-`T_V` ablations, and writes
   `configs/cut_schedule.json`. `python phase2_profile/solve_cut.py --selftest` anchors the solver
   on the paper's own Table 5 (its stage times imply `T_G=13.25 s, T_N=3.16 s, T_V=267 s` → the
   solver returns the paper's `k_b = 51`). Note the paper's own period column (683 s) exceeds
   `max(676, 675)` by ~7 s — expect a fixed per-chunk overhead in your runs too.
5. `phase2_profile/RESULTS.md` — the table of measured rates with the phone's thermal state.

**Gate 2 commands:** `python tools/probe/probe_host.py --cooldown 30` then
`python phase2_profile/solve_cut.py --preset B`.
**Gate 2 criteria:** `T_G`, `T_N`, `T_V` measured ≥ 5× each with cooldown between runs, variance
reported (std dev), cut solved for all presets, zero-copy route decided, and the predicted period
`period(k_b*)` written down **before** Phase 3 exists — that is the prediction Phase 4 tests.

---

## Phase 3 — Two-process pipeline (the bulk of the code)

Deliverables in `phase3_pipeline/`:

1. `mali_cl/*.cl` — INT8 GEMM with epilogue (`C = (acc·s_x·s_w) + b + LoRA delta`), row-softmax,
   RMSNorm-across-heads, LayerNorm, AdaLN apply, RoPE apply, GELU-tanh, patchify/unpatchify.
   - de-fused attention only: QK GEMM → softmax → AV GEMM with a **256-row score block**
   - `char4` operands read directly from the row-major artefact; `arm_dot`/`arm_dot_acc` behind
     `#ifdef cl_arm_integer_dot_product_int8` with a scalar fallback path
2. `pixel_npu/nnapi_graph.cpp` — builds one `ANeuralNetworksModel` per block-range at startup
   (fixed shapes, `PREFER_SUSTAINED_SPEED`, per-channel `channelDim=0` symm quant), with the
   attention/RoPE/GELU decompositions of `docs/03_spec_vs_reality.md`. Same
   `run_blocks(start,count,...)` interface as the GPU executor.
3. `pixel_npu/vae_nnapi.cpp` — decomposed causal 3-D conv decoder + tile schedule; or
   `mali_cl/vae.cl` if Gate 0 says the NPU cannot hold the graph.
4. `coordinator/proc_a.cpp` / `proc_b.cpp` / `ipc.cpp` — Unix domain socket, `SCM_RIGHTS` fd pass
   (route A) or shared `mmap` (route B), eventfd/atomic-flag signalling, **two handoff slots** so
   the GPU can start chunk `i+1` while the NPU still reads chunk `i`.
5. `android_main.sh` — pushes artifacts with `adb push`, sets up the thermal gate, runs a stream of
   `n` chunks per prompt, collects `dumpsys meminfo` before/after and the run log.

**Gate 3 (numerical parity, before any perf claim):** for a fixed seed and prompt, dump block-exit
hidden states from PyTorch (FP32) and from each engine; require cosine ≥ 0.999 per block and
`max|Δ|` below the FP16 noise floor after the 30-block accumulation. Single-block parity first,
then 30 blocks, then the full 6-step chunk.
**Gate 4 (memory):** `dumpsys meminfo` PSS for Proc A + Proc B < 2.2 GiB, and the shared artefact
present **once** (compare `/proc/<pid>/maps` for the artefact file / count distinct `dmabuf` inodes
via `/proc/<pid>/fdinfo`).
**Gate 5 (thermal):** every timed run started at `≤ ambient + 3 °C`, logged.

---

## Phase 4 — Evaluation

1. `phase4_eval/reference_fp32.py` — host FP32 reference using `third_party/DCM` model code +
   `diffusers` in the ComfyUI venv (ZLUDA). Produces `results/ref/<prompt>/frames.npy` (also FP16
   "semantic/NPU-style" and INT8-activation "GPU-style" arms, mirroring paper Table 4's four paths).
2. `phase4_eval/metrics.py` — PSNR, SSIM (skimage), LPIPS (`pip install lpips`) per frame pair +
   aggregate; no-reference VBench optional.
3. `phase4_eval/stream_bench.py` — drives `n` chunks back-to-back over ADB, parses the device log,
   computes steady-state period, videos/hour, SLO attainment at `D = 1.5 ×` NPU-only latency,
   and the `f`-sweep of the deadline factor.
4. `phase4_eval/tables.py` — emits the report tables: configurations (NPU-only / GPU-only / DP /
   Hetero-Tensor / HeteroVDiff), cut ablation (step vs block, solo vs co-run, with/without `T_V`),
   memory table, quality table, and a **prediction-vs-measured** table from Phase 2.

**Gate 6 (report-readiness):** every number in `results/tables.md` traces to a JSON in
`results/raw/`; every configuration ran ≥ 3 times; the cut was **not** re-tuned after seeing the
perf numbers (that would be circular — re-solve only from newly measured rates and say so).

---

## Working without the phone

Three things can be verified on this laptop and should be, because they catch 80 % of the bugs:

1. **Kernel numerics without OpenCL.** The desktop has no OpenCL ICD (only the loader,
   `/c/Windows/System32/OpenCL.dll`; no vendor ICD registered) — so you cannot run Mali kernels
   locally. Instead write each `.cl` file so it also compiles as C via a small shim
   (`phase3_pipeline/tests/cl_host_shim.h` defining `char4`, `get_global_id`, `arm_dot`,
   `arm_dot_acc`) and unit-test it with `clang -x c` against a NumPy reference. That catches
   transposed weights, off-by-one tiling and epilogue scale errors for free.
2. **The whole INT8 pipeline in PyTorch/numpy** — quantise, run 6 steps, compare against FP32. This
   is Gate 1 and needs no device.
3. **The cut solver and pipeline simulator** — `solve_cut.py` operates on a JSON of rates, so the
   coordinator's schedule logic can be unit-tested with synthetic rates (including the boundary
   cases `k* > 180`, `T_V` huge, `T_G == T_N`).
