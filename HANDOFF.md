# HANDOFF — HeteroVDiff on Pixel 6a

*Updated 2026-11-15 (night). This file is the orders sheet: read it first, then
`docs/00_START_HERE.md`. The `T_N` harness landed with this update; last commit before it: `8280bbc`.*

## Mission

Reproduce HeteroVDiff's design on a **Pixel 6a (Tensor G1)**: one INT8 copy of DCM-Wan2.1-1.3B's
base weights shared by two engines, a single block-level cut of the `B = 180` denoising
trajectory, and a two-stage chunk pipeline. Course deliverable = working pipeline + measured
tables + IEEE report; the student owns the hardware rig, slides and paper.

## Where things stand

| | |
|---|---|
| Repo | `C:/Users/harri/Documents/GitHub/HeteroVDiff_Pixel_6a` |
| Device | Pixel 6a attached, serial `25311JEGR11811`, shows as `device` |
| Weights | `weights_raw/DCM_WAN/` — transformer 5.73 GiB FP32, VAE 508 MiB, umT5 5 shards |
| Checkpoint manifest | `configs/block_manifest.json` (verified against the paper) |
| Gate 0 (capabilities) | **done** → `docs/device/RESULTS_gate0.md` |
| Gate 2 (zero-copy) | **pass** on both routes; file-mmap route B preferred |
| **Mali INT8 GEMM** | **done** → `docs/device/RESULTS_gemm_mali.md`; 161.8 ms/block, 877 GOPS, exact vs int32 reference |
| **NPU block graph (`T_N`)** | **partial** → `docs/device/RESULTS_npu_tn.md`; 9/10 GEMMs measured gate-compliant (0.4178 s/block, ≥2.58× `T_G`), `ffn.down` has no compilable graph |
| Gate 1 (artifact) | pending — quantiser not written yet (`phase1_quant/` has only the inspector) |
| Gates 3–5 | pending |
| Blocker | `T_N`, `T_V`, co-run `T_G` unmeasured ⇒ `k_b*` still TBD |

## The two results that changed the project

The tiled Mali kernel runs a Preset-A DiT block in **161.8 ms** (877 GOPS) against the Edge TPU
arithmetic bound of **532 ms**. The GPU is **3.3× faster per block** than the TPU, not 31.6×
slower as the naive kernel suggested. The paper's premise (NPU ≈ 3.4× faster than GPU, therefore
NPU takes the suffix) **inverts on Tensor G1**. Per the locked policy this is reported, not
tuned: `k* = (B·T_N + T_V)/(T_N + T_G)` now lands late, so the GPU takes nearly the whole
trajectory and the NPU's real job becomes covering the VAE decode.

The TPU side then got measured (2026-11-15, `RESULTS_npu_tn.md`) and is **worse than the
arithmetic bound, for structural reasons**: 9 of the block's 10 GEMMs run at 204–289 GOPS
(`ffn.up` 36.62 ms/tile = 289 GOPS), summing to **≥ 0.4178 s/block — already ≥ 2.58× `T_G`** for
70 % of the MACs. Three hard limits: a whole-block single graph **reboots the phone** in
`Compilation_finish`; `ffn.down` (K=8960) **never compiles** (`DEAD_OBJECT` after ~62 s hot,
`OP_FAILED` after 47 s cool); and
**per-output-column weight scales are rejected by `FULLY_CONNECTED`** at `addOperation`, so the
locked quantisation cannot be expressed on the NPU stage. `T_N` stays null until `ffn.down` has a
graph.

Caveat that must survive into the paper: 161.8 ms and the NPU per-GEMM rates are **solo** kernel
rates. The NPU rates are now gate-compliant (cool start, TPU 29.0 °C, capture `…_160146`) and a cool
start is **1.6 % slower** than the earlier hot ones, i.e. nothing was throttled. Every stage rate in
the cut must still come from a co-run measurement. Reaching the idle floor took 17 min with the
device network off — use `python tools/npu/cool_rerun.py`, never a bare timed run.

## Ordered next actions

1. ~~**Cool re-run of `T_N`**~~ — **done 2026-11-15 16:00:42**: capture `…_160146`, TPU 29.0 °C start
   against a 29.1 °C gate, gate-compliant table in `RESULTS_npu_tn.md`. Reusable tool:
   `python tools/npu/cool_rerun.py` (waits for the floor, then runs the harness under it). Note:
   with the Play Store background service running the TPU zone sat at 44–48 °C and the gate was
   unreachable — kill device network for the run.
2. **`ffn.down` path** — the down projection has no compilable FC. Pick and measure one of:
   chunked-K FC + `ADD` (2× K=4480 or 4× K=2240) on the TPU, or the down projection on Mali
   (OpenCL, where the tiled kernel already runs this shape at 890 GOPS per `RESULTS_gemm_mali.md`).
   This is now the *only* thing standing between the 0.4178 s lower bound and a real `T_N`.
3. **Per-column scales vs the NPU stage.** `FULLY_CONNECTED` only takes per-tensor weight scales.
   Either the NPU graph uses per-tensor scales (measure the Gate 1/4 parity cost) or the NPU is
   restricted to shapes with an equivalent path. This touches `design_locks.json` → student's call.
4. **`T_V` — VAE decode.** No `CONV_3D` in NNAPI: either decompose the causal 3D convs into
   shifted 2D `CONV_2D` taps on the TPU, or decode on Mali in OpenCL. Measure both, report both.
5. **Co-run `T_G`.** Two processes, one file-mapped artifact, GPU on blocks `[0,k)` while the
   NPU runs blocks `[k,180)` + decode. This is the number the cut is allowed to use.
6. `python phase2_profile/solve_cut.py --Tg … --Tn … --Tv …` → commit the measured `k_b*`.
7. Phase 1 (quantiser + Gate 1) is still outstanding and needs no phone — it can run in parallel
   or while the phone cools: `docs/05_phase_plans.md`.
8. Phase 3 executors, Phase 4 tables, report.

## Decisions already locked (do not reopen without saying why)

- Backbone: **DCM-Wan2.1-1.3B**, `N = 6`, `L = 30`, `B = 180`; expert switch `κ = 3`.
- Quantisation: **per-output-column symmetric INT8**, one FP32 scale per column, base weights only;
  bias/norm/AdaLN/embeddings stay FP16.
- Detail expert stored **factorised** (rank-128 `A`,`B` + its own timestep block), never merged.
  ⚠️ measured cost of FP16 factors: **+226 ms per detail block** vs **+41 ms** if the factors were
  INT8 — a decision to put to the student, not to change quietly (`RESULTS_gemm_mali.md` §6).
- One **block-level** cut, one handoff per chunk, `k_b` from co-run rates with `T_V`.
- Layout: **row-major `[N_out, K]` INT8** — verified to be exactly what both `char4`+`arm_dot`
  and NNAPI `FULLY_CONNECTED` want.
- The NPU stage is a **NNAPI graph**, not custom kernels.
- Attention **de-fused** (QK → softmax → AV), 256-row score block.
- No fabricated device numbers: unknown ⇒ `null` + "TBD (blocked: …)".

## Kernel facts worth not rediscovering

- `clFinish` is **not** a completion barrier on Mali r54p2 — close every timing with a blocking read.
- Global size is `ceil(N/BN)·(BN/TN)`; the `round_up` form corrupts memory and surfaces only as
  `CL_EXEC_STATUS_ERROR (-14)`.
- **32 int accumulators maximum** per thread; 64 spills (112–144 B/thread) and costs 4.5×.
- Predicate the epilogue — a `break` in an unrolled loop spills the accumulators.
- `arm_dot_acc` is a real SDOT; measured ceiling **1.77 TOPS**.

NNAPI / Edge TPU (paid for 2026-11-15, evidence in `RESULTS_npu_tn.md`):

- **A model needs `identifyInputsAndOutputs`.** Operands + `addOperation` alone compile to
  `model_finish → BAD_DATA`. That was the first hour of the `T_N` harness.
- **`getSupportedOperationsForDevices` requires a FINISHED model** — called before
  `model_finish` it returns `BAD_STATE` and tells you nothing.
- **A 10-op chained block graph reboots the phone** inside `Compilation_finish` (uptime resets).
  It validates and reports full support first, so validation is not a viability test.
- **`ffn.down` (K=8960) never finishes compiling** → `DEAD_OBJECT` after ~62 s, reproducible fresh.
- **`FULLY_CONNECTED` rejects `TENSOR_QUANT8_SYMM_PER_CHANNEL`** (`addOperation` → `BAD_DATA(4)`).
- `getDuration(DURATION_ON_HARDWARE)` reports ~6 µs for a 0.9 GMAC GEMM → unusable; time the
  synchronous `execute`.

## Definition of done for the course project

1. Artifact + quantiser + parity report (Gate 1) committed.
2. Device probe + measured rates + solved cut (Gate 2) committed. ← partially: GEMM done
3. Two-process pipeline streaming ≥10 chunks on the phone without manual intervention (Gates 3–5).
4. `results/tables.md`: configurations, cut ablation, memory, quality (PSNR/SSIM/LPIPS), SLO sweep,
   plus a prediction-vs-measured table from Phase 2.
5. Report-ready list of deviations from the paper, each with the measurement that forced it.
