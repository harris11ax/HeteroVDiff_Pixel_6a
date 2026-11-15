# HeteroVDiff — paper digest (only what you need to reproduce it)

Source: `2610.16244v1.pdf` (HeteroVDiff, Shen et al., UMD). Everything here is from the paper;
anything I measured myself is marked **[measured]**.

## The claim in one line

Serve a multi-expert video diffusion transformer across phone **NPU + GPU** from **one stored
INT8 copy of the base weights**, cut the flattened denoising trajectory **once at a block
boundary**, and overlap GPU-prefix of video *i+1* with NPU-suffix+decode of video *i* →
**1.35× (49 frames) to 1.50× (17 frames)** stream throughput over NPU-only on a Galaxy S25,
at ≤ 0.17 pp VBench quality loss.

## Workload

- DCM-Wan2.1-1.3B, `N = 6` denoising steps × `L = 30` DiT blocks = `B = 180` block executions.
- 832×480, **49 frames**, latent `C_z=16, F_z=13, H_z=60, W_z=104`, patch `p=2` → `M = 20,280` tokens.
- Timesteps `[1000, 992, 982, 949, 905, 810]`; detail expert when `t ≤ 978` ⇒ `κ = 3`, so
  steps 0–2 semantic / 3–5 detail; expert switch at block `κ·L = 90`.
- Semantic expert = shared base (1,419 M params); detail expert = same base + rank-128 LoRA on
  attention projections (94.4 M) + its own timestep-embedding block (25.6 M).

## Measured rates that drive the design (Galaxy S25, cold, screen off, external cooler)

| | per DiT block | VAE / video | video total |
|---|---|---|---|
| NPU solo | 3,352 ms | 308 s | 916.7 s |
| NPU co-run | 3,101 ms | 275 s | 825 s |
| GPU solo | 12,633 ms | — | — |
| GPU co-run | 12,800 ms | — | — |

Precision table at `M = 20,280` (Table 2): GPU **INT8 12.6 s** vs FP16 42.3 s (3.37×) vs FP32
37.8 s; NPU **FP16 3.36 s** vs INT8 9.95 s (2.96×) vs FP32 3.95 s (1.17×).
⇒ the two engines want **opposite** formats; that is why the shared artifact is INT8 (GPU's
fast path) and the NPU **dequantises to FP16 per block** into a 52.5 MiB scratch.

## The three mechanisms

1. **One INT8 artifact, one region per DiT block** (paper §4.2, Eq. 1–2)
   `s_j = max_i |W_ij| / 127`, `Q = clip(round(W/s_j), [-127,127])`, stored in the NPU's 32×32
   HMX tile order; the last term of Eq. 2 packs 4 consecutive input values of a fixed output
   column into 4 contiguous bytes = exactly the `char4` operand Adreno's dp4a takes. Base
   artifact total 1,423 MiB; per block **47.3 MiB**; attention base 541.4 MiB, FFN pair
   787.5 MiB, scales 2.6 MiB, rank-128 factors 91.5 MiB. Regions must be per-block because
   **no cross-runtime allocation exists**: gralloc refuses a BLOB at exactly 2^29 B, and 511 MiB
   maps into both QNN and OpenCL while 1024 MiB takes the device down.
2. **One block-level cut in the flattened trajectory** (§4.3, Fig. 4)
   GPU `[0, k_b)`, NPU `[k_b, B)` + decoder. One handoff per video (hidden state `X_{t,l}` +
   patchified latent `z̄_t`, 130 MB). Because `k_b` need not be a multiple of `L`, the NPU can
   resume mid-step; the NPU block kernel takes `start_block, count`.
   `period(k_b) = max(k_b·T_G, (B-k_b)·T_N + T_V)`; `k* = (B·T_N + T_V)/(T_N + T_G)`; take the
   integer neighbour with the smaller period. Deployed cuts: `k_b = 51, 54, 59, 61` for
   49/33/25/17 frames.
3. **Platform limits measured first** (§3): max buffer, which buffers both runtimes can map,
   per-graph activation ceiling (QNN: 2 GiB), and how co-execution changes each engine's rate
   (NPU speeds up 10→19 %, GPU slows down slightly, as clips shorten).

## Cut placement ablation (Table 5, 49 frames) — the table your report should reproduce

| policy | k_b | GPU s | NPU s | idle s | period s |
|---|---|---|---|---|---|
| step boundary before | 30 | 431 | 765 | 334 | 768 (+12.5 %) |
| step boundary after | 60 | 782 | 650 | 132 | 782 (+14.5 %) |
| block rates, no `T_V` | 34 | 443 | 723 | 280 | 726 (+6.3 %) |
| **solo** rates | 57 | 747 | 655 | 92 | 752 (+10.1 %) |
| **co-run** rates + `T_V` | **51** | 676 | 675 | **1** | **683** |

Lesson to reproduce: solo rates over-credit the GPU by 6 blocks; ignoring `T_V` over-credits the
GPU by 17 blocks. Both errors are systematic, not noise. `T_V` drifted 186→270 s between
measurement days, so **re-measure `T_V` for the day of the run**.

## Other constraints worth carrying over

- GPU keeps the inter-block hidden state in **FP16** and quantises each GEMM input with a
  per-block, per-GEMM scale calibrated at `max|x|/127` under the real timestep schedule and
  AdaLN modulation (§4.4). NPU keeps the residual stream in **FP32** (it grows to ~25 in
  magnitude over 30 gated adds) and narrows to FP16 only at GEMM operands.
- Attention must be **de-fused** (QK GEMM → row softmax → AV GEMM): a single fused INT8 kernel
  was killed before completing at deployment length (Appendix B.2). Keeping one 256-row block
  of scores bounds memory at `O(M)` not `O(M²)`.
- Working set: ~1.9 GiB steady state for the whole system (weights 1.47 + scratch 0.078 +
  buffers ~0.4). Handoff 130 MB/video.
- Output quality (Table 4, VBench Quality score): FP32 83.19, NPU path 83.25, GPU path 82.74,
  HeteroVDiff(k_b=51) 83.02 → the shared INT8 artifact costs ≈ nothing perceptually.

## Evaluation protocol (reproduce the shape, not the numbers)

- 10 content-diverse prompts, generated back to back, **decoding inside the pipeline**;
  steady-state period after the initial fill.
- Configurations: NPU-only (1.0× reference), GPU-only (DiT on GPU + decode on NPU, because
  the vendor GPU backend has no Conv3d), DP (naive whole-video per engine), Hetero-Tensor
  (per-GEMM column split), **HeteroVDiff** (one block cut).
- SLO: per-video deadline `D = 1.5 ×` NPU-only single-video latency; report videos/hour
  **within** the SLO. DP collapses to the NPU-only rate because its GPU-arm videos miss `D`.
- Memory: count tensor sizes, not RSS pages (RSS moves with page cache/compressed swap).
- Cold start each timed run; screen off; plugged in; external cooling.
