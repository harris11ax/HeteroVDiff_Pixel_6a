# HANDOFF — HeteroVDiff on Pixel 6a

*Updated 2026-11-15 (evening). This file is the orders sheet: read it first, then
`docs/00_START_HERE.md`. Last commit before this update: `bb05143`.*

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
| Gate 1 (artifact) | pending — quantiser not written yet (`phase1_quant/` has only the inspector) |
| Gates 3–5 | pending |
| Blocker | `T_N`, `T_V`, co-run `T_G` unmeasured ⇒ `k_b*` still TBD |

## The one result that changed the project

The tiled Mali kernel runs a Preset-A DiT block in **161.8 ms** (877 GOPS) against the Edge TPU
arithmetic bound of **532 ms**. The GPU is **3.3× faster per block** than the TPU, not 31.6×
slower as the naive kernel suggested. The paper's premise (NPU ≈ 3.4× faster than GPU, therefore
NPU takes the suffix) **inverts on Tensor G1**. Per the locked policy this is reported, not
tuned: `k* = (B·T_N + T_V)/(T_N + T_G)` now lands late, so the GPU takes nearly the whole
trajectory and the NPU's real job becomes covering the VAE decode.

Caveat that must survive into the paper: 161.8 ms is a **solo** kernel rate. Every stage rate in
the cut must come from a co-run measurement.

## Ordered next actions

1. **`T_N` — token-tiled NPU block graph.** One NNAPI graph per (block, token-tile) at M=384,
   4 tiles for Preset A, chained inside one Execution where the compiler allows it. Measure
   s/block. This is the missing half of the cut. Traps already known:
   `docs/device/RESULTS_gate0.md` §"Getting an NNAPI graph to validate".
2. **`T_V` — VAE decode.** No `CONV_3D` in NNAPI: either decompose the causal 3D convs into
   shifted 2D `CONV_2D` taps on the TPU, or decode on Mali in OpenCL. Measure both, report both.
3. **Co-run `T_G`.** Two processes, one file-mapped artifact, GPU on blocks `[0,k)` while the
   NPU runs blocks `[k,180)` + decode. This is the number the cut is allowed to use.
4. `python phase2_profile/solve_cut.py --Tg … --Tn … --Tv …` → commit the measured `k_b*`.
5. Phase 1 (quantiser + Gate 1) is still outstanding and needs no phone — it can run in parallel
   or while the phone cools: `docs/05_phase_plans.md`.
6. Phase 3 executors, Phase 4 tables, report.

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

## Definition of done for the course project

1. Artifact + quantiser + parity report (Gate 1) committed.
2. Device probe + measured rates + solved cut (Gate 2) committed. ← partially: GEMM done
3. Two-process pipeline streaming ≥10 chunks on the phone without manual intervention (Gates 3–5).
4. `results/tables.md`: configurations, cut ablation, memory, quality (PSNR/SSIM/LPIPS), SLO sweep,
   plus a prediction-vs-measured table from Phase 2.
5. Report-ready list of deviations from the paper, each with the measurement that forced it.
