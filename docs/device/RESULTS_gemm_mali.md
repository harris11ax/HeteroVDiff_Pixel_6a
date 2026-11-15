# Tiled INT8 GEMM on Mali-G78 — measured

**The question this answers.** `docs/device/RESULTS_gate0.md` left the Mali path at 10.4 GOPS
(naive, one thread per output) against 285 GOPS for the Edge TPU: a 31.6× gap that would make
Process A a formality (`k_b* ≈ 7`, gain ≈ 1.03×). Everything downstream — whether the GPU is a
real pipeline stage, and where the cut lands — depends on how much of that gap kernel
engineering closes. This file is the measurement, not a model.

Sources: `tools/android/gemm/gemm_bench.cpp`, `tools/android/gemm/kernels/gemm8.cl`, driver
`tools/gemm/gemm_host.py`, raw captures `docs/device/gemm_mali_*.json`, analysis
`phase2_profile/gemm_analysis.py`.

| run | what | artifact |
|---|---|---|
| gated, 3 variants + reference | correctness, tile sweep, block time, LoRA cost | `gemm_mali_20261115_152103.json` |
| gated, `--mode peak` | dot-throughput ceiling | `gemm_mali_20261115_152337.json` |

Thermal gate on both: start at `max(TPU, skin, VIRTUAL-SKIN) ≤ battery + 3 °C`
(before: TPU 32.0, skin1 30.5, skin2 31.3, battery 29.4). The gated run ended at TPU 52.0 °C —
the *start* is controlled, the end is not, and that is stated rather than hidden.

## 1. Correctness comes first, and it is exact

Every kernel is checked against `gemm8_ref`, a one-thread-per-output int32 GEMM, on the same
`A` and `W`. Integer arithmetic is exact, so the outputs must match **bit-for-bit** — no
tolerance. Two shapes: `M=384` (clean) and `M=1530` (exercises the row-padding path, since
`ceil(1530/64)·64 = 1536`).

```
VERIFY_SUMMARY|fails=0
VERIFY|tiled_64x64x32_t8x4|M=1530|N=256|K=1536|MATCH|mismatch=0|nonzero=391680
```

This matters because three of the bugs found below (`§5`) fail *silently*: they produce
plausible-looking numbers, or a bare `CL_EXEC_STATUS_ERROR (-14)` on the next command with no
message. A kernel that runs is not a kernel that is right.

## 2. The ceiling: `arm_dot_acc` is a real SDOT, and it is 1.77 TOPS

Before tuning anything, measure what the instruction can do. Three probes with an identical
inner loop (`8×8` operand pairs, 8 accumulator chains, 256 MACs per iteration, one global load
per iteration so nothing can be hoisted):

| probe | instructions per 256 MACs | measured (gated run) |
|---|---|---|
| `dg_peak_dot` — 64 × `arm_dot_acc` | 64 | **827.8 GMAC/s = 1655 GOPS** |
| `dg_peak_mac4` — 256 × `mad24` | 256 | 124.2 GMAC/s = 248 GOPS |
| `dg_peak_mad1` — 64 × `mad24` | 64 | 15.5 GMAC/s = 31 GOPS |

`dot / mac4 = 6.7×`. So the builtin compiles to genuine SDOT instructions and the arithmetic
ceiling is **1655 GOPS**. Nameplate cross-check: 20 CUs × 16 lanes × 4 int8-MAC × 848 MHz =
1.085 TMAC/s = 2170 GOPS, so the probe reaches **76%** of the driver-reported ceiling — the
missing quarter is the one global load per iteration plus warp-scheduling overhead. (An
ungated repeat measured 886 GMAC/s = 1772 GOPS, so treat the ceiling as ~1.7 TOPS ± 7%.)

Note the build warning that comes with it:

```
warning: unknown OpenCL extension 'cl_arm_integer_dot_product' - ignoring
```

The pragma is ignored *and the builtin still works*. Do not read that warning as "the dot
product is unavailable" — the probe above is the evidence that it is real.

## 3. The tile sweep: where the 31.6× goes

Preset A token counts (`M = 1530`), one GEMM per shape, 3 dispatches back to back, one blocking
read at the end (`gemm_mali_20261115_152103.json`):

| variant (BM×BN×BK, thread tile) | private mem | attn 1536² | ffn_up 1536→8960 | ffn_down 8960→1536 |
|---|---|---|---|---|
| `ref_naive` (1 thread/output) | 0 | ~10 GOPS | ~10 GOPS | ~10 GOPS |
| `64×64×16`, tile 8×8 | 144 B/thr | 192 | 188 | 141 |
| `64×64×16`, tile 8×4 | **0** | 586–783 | 868–905 | 772–781 |
| `64×64×32`, tile 8×4 | **0** | **751** | **869** | **890** |

GOPS = 2·M·N·K / t, the same convention as the Gate-0 numbers. Two things dominate:

1. **Accumulator spilling.** 64 live `int` accumulators (8×8) do not fit; the compiler spills
   to private memory (112–144 B/thread reported by `CL_KERNEL_PRIVATE_MEM_SIZE`) and throughput
   drops **4.5×** — the 8×8 and 8×4 variants run the *same* arithmetic. 32 accumulators (8×4) is
   the largest tile that stays in registers, and it is where the curve peaks.
2. **Local-memory depth.** `BK=32` beats `BK=16` by ~5% at the block level: fewer barriers and
   fewer global staging rounds.

`8×4` at `64×64×32` is the deployed configuration: 16.6 KiB of local memory per work-group
(two padded `char4` slices), no private memory, 877 GOPS effective on a full block.

## 4. The number that decides the project

One DiT block at Preset A = 8 attention GEMMs (`K=N=1536`) + FFN up (`1536→8960`) + FFN down
(`8960→1536`) = **70.99 GMAC = 142.0 GFLOP**. Timed as a 10-dispatch sequence with the kernels
built once (not as a sum of separately-warmed launches):

| | measured | vs the bounds in RESULTS_gate0 §"what that predicts" |
|---|---|---|
| tiled Mali, `64×64×32`, tile 8×4 | **161.8 ms / block** (877 GOPS = 53% of the 1655 GOPS dot ceiling) | TPU bound 532 ms → **3.29× faster** |
| tiled Mali, `64×64×16`, tile 8×4 | 168.6 ms / block (842 GOPS) | 3.16× faster |
| tiled Mali, `64×64×16`, tile 8×8 (spills) | 732.7 ms / block (194 GOPS) | 0.73× — slower than the TPU |
| naive Mali (Gate 0) | 14.6 s / block | 0.04× |
| 1 TOPS hypothetical | 152 ms / block | the measured kernel reaches **0.88 TOPS** |

Sum-of-parts cross-check (8 × 9.62 + 48.47 + 47.33 ms from the per-GEMM sweep) = 172.7 ms,
about 7% above the sequence measurement, which is the expected per-launch overhead when each
GEMM is timed in isolation.

**Consequence.** The GPU is not the slow engine on this SoC — it is **3.3× faster per block than
the Edge TPU bound**. The paper's premise (NPU ≈ 3.4× faster than GPU, so the NPU must take the
suffix) **inverts on Tensor G1**: `T_G ≈ 0.16 s/block` against `T_N ≈ 0.53 s/block`. Under the
locked policy that is the finding to report, not something to tune away — `k* = (B·T_N + T_V)/
(T_N + T_G)` then lands late (`k_b* → 179` as `T_V` grows), i.e. the GPU takes almost the whole
trajectory and the NPU survives as the VAE host, not as a co-equal block engine. The cut is
still worth shipping, but the *reason* it helps changes: it is no longer "offload to the faster
engine", it is "cover the decoder with GPU blocks".

⚠️ `T_G` here is a **solo** kernel rate. The locked rule is that stage rates come from
co-running measurements; the co-run `T_G`, the token-tiled `T_N` and `T_V` are **TBD** — see §7.

## 5. Traps that cost runs (all now fixed in the harness or kernel)

1. **`clFinish` is not a completion barrier on this driver.** It returned in 0.069 ms for
   3.6 GMAC (≈4 TOPS, above the 1.09 TOPS nameplate). Every timing must be closed with a
   blocking read. This silently produced impossible GOPS until the probe caught it.
2. **The global NDRange is `ceil(N/BN)·(BN/TN)`, not `round_up(N, lx)`.** The wrong form
   dispatched ~8× too many work-groups, which wrote past the output buffer; the driver reports
   that only as `CL_EXEC_STATUS_ERROR (-14)` on the next blocking read, with nothing in
   `logcat`. Symptom to recognise: every variant "fails" but `enq=0`.
3. **`break` in an unrolled epilogue spills the accumulators.** A bounds `break` prevents
   unrolling, the accumulator array becomes dynamically indexed, everything lands in private
   memory, and the kernel dies with `-14`. Use predicated `if`s.
4. **Local-memory staging must use the work-group tile base, not the thread's offset.** Staging
   from `m0` (thread base) has all 64 threads stage different windows into the same shared
   buffer: results are wrong but plausible, and at `M=1530` the last tiles read past `A` and
   take the queue down.
5. **64 live accumulators do not fit** (§3). `CL_KERNEL_PRIVATE_MEM_SIZE` is the quick check —
   anything above 0 on this kernel means the tile is too big.
6. **A 2-D local work-group size is fine**, including `(8,8)`: `dg_2d` and a 1×1 dot kernel pass
   at `(8,8)`, `(16,4)`, `(64,1)` and driver-chosen sizes. The failure mode was never the
   dispatch shape.

## 6. The detail expert's LoRA delta: a design lever worth 226 ms/block

The detail-expert path adds a rank-128 delta to the eight attention projections per block. Three
ways to compute it, measured on the same block sequence:

| how | ms / block (steps 3–5) | note |
|---|---|---|
| base INT8 GEMM only (semantic steps) | **161.8** | |
| INT8 GEMM + precomputed delta added in the epilogue | **161.5** | the add is free; delta computed by 2 FP16 GEMMs |
| INT8 GEMM + naive rank-R dot per output element | 634.7 | rejected: `U`/`Bl` reads have no reuse |
| the 2 FP16 GEMMs themselves, FP16 factors (229.0 MiB) | +226 | 16.0 + 12.2 ms per projection × 8 |
| the 2 GEMMs if the factors were stored INT8 (114.5 MiB) | **+41** | 2.83 + 2.30 ms per projection × 8 |

So: **fuse the delta add, never fuse the delta *computation*** (3.9× penalty), and note that
storing the factors INT8 rather than FP16 — half the bytes, 5.5× cheaper delta — is worth a
decision. `configs/design_locks.json` currently says FP16 factors; the measured cost of that
choice is **+185 ms on every detail block**, i.e. `T_G ≈ 0.39 s` for steps 3–5 against
`T_G ≈ 0.16 s` for steps 0–2. That is a lock to revisit with the student, not to change quietly.

## 7. What is still TBD, and why

| quantity | state |
|---|---|
| `T_G` (co-run, s/block) | **TBD (blocked: solo kernel rate measured; co-run measurement needs the two-process coordinator)** |
| `T_N` (token-tiled NPU block graph, s/block) | **TBD (blocked: the 384-token tile graph is not built; 532 ms is the arithmetic bound from Gate 0, not a measurement)** |
| `T_V` (VAE decode, s/chunk) | **TBD (blocked: no `CONV_3D` in NNAPI; the decomposed-2D-conv and Mali-OpenCL paths are unbuilt and unmeasured)** |
| `k_b*` | **TBD (blocked on the three rates above)** |

What is *not* TBD: the Mali INT8 executor is real, exact against an integer reference, and fast
enough that Process A is a genuine stage. The kernel is now the same order as the Edge TPU
(877 vs 285 GOPS) instead of 31.6× behind it.

## 8. Reproduce

```
python tools/gemm/gemm_host.py --build                       # NDK, no Gradle
python tools/gemm/gemm_host.py --mode all --iters 3 \
       --cooldown 32.0 \
       --variants ref_naive,tiled_64x64x16_t8x8,tiled_64x64x16_t8x4,tiled_64x64x32_t8x4
python tools/gemm/gemm_host.py --mode peak --cooldown 32.0    # the 1.77 TOPS ceiling
python phase2_profile/gemm_analysis.py docs/device/gemm_mali_20261115_152103.json
```

Kernel source hash and binary hash are recorded in every JSON, so a GOPS number can always be
traced to the exact `.cl` that produced it.
