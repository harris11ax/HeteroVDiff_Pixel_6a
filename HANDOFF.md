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
| **NPU block graph (`T_N`)** | **measured as a sum** → `docs/device/RESULTS_npu_tn.md`; 9 GEMMs gate-compliant (0.4178 s) + `ffn.down` chunked 7×K=1280 (0.2580 s) = **0.6758 s/block**, 210 GOPS |
| **VAE decode (`T_V`)** | **both branches measured** → `docs/device/RESULTS_vae_tv.md`; TPU (36 causal convs as `kT` shifted `CONV_2D` taps) **10.554 s/chunk**, 969.7 GOPS; Mali (`conv8_taps`, implicit GEMM, one dispatch per class) **12.256 s/chunk**, 417.5 GOPS, 7/13 classes exact vs the int32 reference. **TPU 1.16× faster** — the decode ordering is the opposite of the block's. Mid-block attention (32.4 GMAC) still unmeasured ⇒ `T_V` is a lower bound on both |
| Gate 1 (artifact) | **INT8 base + verifier done** → `docs/device/RESULTS_gate1.md`; 1,391,984,640 B INT8 (exact), 300/300 GEMMs cosine ≥ 0.999922, coverage 1317/1317 tensors. **Outstanding: end-to-end PSNR/SSIM leg (blocked: no torch/diffusers on this host)** |
| Gates 3–5 | pending |
| **Co-run `T_G` / `T_N`** | **measured** → `docs/device/RESULTS_corun_tg.md`; both engines executing at once, 99.8 % of GPU-time overlapped NPU executes. GPU base block **0.1474 s** (no slowdown — it measured 8 % *faster* than solo), NPU 9-GEMM round **0.3904 s** (±4 % per GEMM). `T_N/T_G = 4.40×` |
| **`k_b*`** | **solved = 160** → `configs/rates_preset_a.json` + `phase2_profile/solve_cut.py`; period 23.58 s, 5.40× vs NPU-only, GPU `[0,160)` + NPU `[160,180)` + decode |
| Blocker | mid-block VAE attention has no measured rate on either engine (`T_V` is a lower bound); `ffn.down` is not in the contended number. `T_V` 10.554 s (TPU) vs 12.256 s (Mali) — TPU-favourable |

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
locked quantisation cannot be expressed on the NPU stage.

`ffn.down` was then closed (2026-11-15 17:48, `RESULTS_npu_tn.md` §4): the compiler's limit is the
**reduction dimension** — K ≤ 1664 compiles in ~0.5 s, K ≥ 1792 grinds for 46–59 s and returns
`OP_FAILED` — so the down projection has no graph at K=8960 *or* at any 2-way/4-way split. It does
run as 7 × FC`[1536,1280]` + 6 × `ADD`, one compilation per op (2.57 s of compile), at
**0.2580 s/block (163.9 GOPS) — 5.45× the Mali time for the same shape (47.33 ms)**. `T_N` is
therefore **0.6758 s/block** and `T_N/T_G = 4.18×`: the GPU is 4.2× faster per block, and the down
projection belongs on Mali whichever way the cut lands. Both branches are measured; the choice is
the student's.

Caveat that must survive into the paper: 161.8 ms and the NPU per-GEMM rates are **solo** kernel
rates, and a cool start is **1.6 % slower** than the earlier hot ones, i.e. nothing was throttled.
The co-run then closed the gap the cut actually needs (`RESULTS_corun_tg.md`): with both engines
executing at once the GPU base block costs **0.1474 s** and the NPU's 9-GEMM round **0.3904 s** —
neither engine loses its solo rate (the GPU measured 8 % *faster* under load). `T_G`/`T_N` from the
co-run give **`k_b* = 160`** (period 23.58 s, 5.40× vs NPU-only). Reaching the idle floor took 17 min
with the device network off — use `python tools/npu/cool_rerun.py`, never a bare timed run.

## Ordered next actions

1. ~~**Cool re-run of `T_N`**~~ — **done 2026-11-15 16:00:42**: capture `…_160146`, TPU 29.0 °C start
   against a 29.1 °C gate, gate-compliant table in `RESULTS_npu_tn.md`. Reusable tool:
   `python tools/npu/cool_rerun.py` (waits for the floor, then runs the harness under it). Note:
   with the Play Store background service running the TPU zone sat at 44–48 °C and the gate was
   unreachable — kill device network for the run.
2. ~~**`ffn.down` path**~~ — **done 2026-11-15 17:48, both branches measured**. Single FC has no
   compilable graph at K=8960/4480/2240/1792 (compiler K ceiling: 1664 OK in ~0.5 s, 1792 fails
   after 48.7 s). Measured: Mali **47.33 ms/block** vs Edge TPU 7×K=1280+6×ADD
   **258.0 ms/block** ⇒ 5.45× in Mali's favour. New modes: `downsep`, `kcompile`, `ksweep`,
   `diag` (see `RESULTS_npu_tn.md` §4). `T_N` = **0.6758 s/block**.
3. **Per-column scales vs the NPU stage — cost now measured, decision still yours.** `FULLY_CONNECTED`
   takes per-tensor weight scales only. Measured on the real weights (40 GEMM instances, blocks
   0/5/15/29): per-tensor costs **2.15× the mean GEMM output error** (3.01 % vs 1.40 %, worst 5.44×)
   and fails the Gate-1 `cosine ≥ 0.999` weight bar on 5/40 instances (worst 0.996892) where
   per-column fails none; the penalty tracks column-scale spread at **r = 0.805**, i.e. it is a
   weight property, not a shape property — so "restrict the NPU to shapes that quantise the same"
   is not supportable, and the cheap subset (`attn2.to_k`+`to_v`) is only **10.2 % of block MACs**.
   → `docs/device/RESULTS_scales.md`, `phase1_quant/scale_parity.py`. Touches `design_locks.json`.
4. **`T_V` — VAE decode.** ~~TPU branch done~~ **both branches now measured 2026-11-15**:
   TPU `CONV_2D` taps **10.554 s/chunk** (969.7 GOPS, gate-compliant start) vs Mali
   `conv8_taps` **12.256 s/chunk** (417.5 GOPS, TPU 26.0 °C start, 7/13 classes exact) ⇒
   **the NPU is 1.16× faster at the decode**, the opposite ordering to the DiT block.
   Still open: (a) the mid-block 3-D attention (32.4 GMAC, 0.6 % of MACs) — no measured
   rate on either engine, so `T_V` is a lower bound both ways; (b) the Mali conv runs at
   48 % of its own GEMM rate (378–486 vs 877 GOPS) because the activation tile is re-staged
   per tap — a halo-staged tile is the next measurement, not a model.
5. ~~**Co-run `T_G`**~~ — **done 2026-11-15 19:48** (`RESULTS_corun_tg.md`, capture `corun_tg_20261115_194809`):
   two processes, one `adb shell`, 99.8 % overlap proven from `CLOCK_MONOTONIC` stamps. GPU base block
   **0.1474 s** co-run vs 0.1618 solo (it got *faster*, not slower — contention below the noise floor),
   NPU 9 GEMMs **0.3904 s** co-run (±4 % per GEMM vs solo) ⇒ `T_N` co-run = 0.6484 s (downsep part still
   solo). New tool: `python tools/corun/co_run.py` (`npu_block corun` holds 9 compilations and loops).
6. ~~`python phase2_profile/solve_cut.py --Tg … --Tn … --Tv …` → commit the measured `k_b*`~~ —
   **done**: `configs/rates_preset_a.json` carries the measured pair; `k_b* = **160**` (period 23.58 s,
   5.40× vs NPU-only; 158 on the conservative solo pair, 162 with the Mali decode). The cut is sharp —
   step-level `k_b` (multiple of 30) costs 27 %, and dropping `T_V` from the solve costs 35 %.
7. ~~Phase 1 (quantiser + Gate 1)~~ — **INT8 base artifact done 2026-11-15 20:5x**:
   `phase1_quant/quantize_int8.py` (40.1 s) + `phase1_quant/verify_artifact.py` (61 s, independent:
   re-reads the bytes, re-quantises the source, demands byte identity). 30 regions × 46,399,488 B =
   **1,391,984,640 B INT8, delta 0** vs Gate 1; 44.25 MiB/region (limit 47.29); MAC-weighted weight
   error **0.93 %**, cosine 0.999956, worst GEMM `ffn.net.2` 1.25 %/0.999922 — all 300 ≥ 0.999;
   max saturation 0.070 %; FP16 remainder 54,024,320 B over 525 tensors (0.028 % worst); coverage
   1317/1317. → `docs/device/RESULTS_gate1.md`. Bug caught en route: a suffix filter had dropped all
   30 per-block `scale_shift_table` (276,480 params) while still printing PASS — classification is
   now by role, with a coverage assertion in both scripts.
   **Still open in Gate 1:** (a) the end-to-end FP32-vs-INT8 PSNR ≥ 28 dB / SSIM ≥ 0.92 leg —
   blocked, this host has no torch/diffusers; (b) deliverables 2–4 (`expert_factors.py` — expects
   240,089,216 B FP16, `vae_export.py`, `calibrate.py`) are not written.
8. Mid-block VAE attention (32.4 GMAC) is the last unmeasured rate; it is the only thing keeping `T_V`
   a lower bound. Needs a GEMM rate at 6120 tokens × 384 ch on both engines.
9. Phase 3 executors, Phase 4 tables, report.

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
- **`CONV_2D` (VAE decode) compiles only at N=1 above ~2.4 MB/frame**: N=3 at 0.78 MB/frame
  compiles in 141.6 ms; N=3/N=5 at 1.4 MB and N=3/N=9 at 6.3 MB fail after 31–36 s, while the
  same 37.6 MB tensor compiles in 263 ms at N=1. Support says `delegated=1` either way.
  `CONV_2D` also reaches **685–1842 GOPS** vs 285 GOPS for FC on the same silicon.
- `getDuration(DURATION_ON_HARDWARE)` reports ~6 µs for a 0.9 GMAC GEMM → unusable; time the
  synchronous `execute`.

## Definition of done for the course project

1. Artifact + quantiser + parity report (Gate 1) committed.
2. Device probe + measured rates + solved cut (Gate 2) committed. ← partially: GEMM done
3. Two-process pipeline streaming ≥10 chunks on the phone without manual intervention (Gates 3–5).
4. `results/tables.md`: configurations, cut ablation, memory, quality (PSNR/SSIM/LPIPS), SLO sweep,
   plus a prediction-vs-measured table from Phase 2.
5. Report-ready list of deviations from the paper, each with the measurement that forced it.
