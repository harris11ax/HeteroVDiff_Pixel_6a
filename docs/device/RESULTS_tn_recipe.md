# P3.7c — what the *bit-exact* NPU stage costs, and it is exact at 448/448 tiles

**Status: MEASURED.** P3.7b localized Gate 3's failure to the M row tile and mapped the only
shapes this HAL returns bit-exactly (`N ∈ {128, 384, 768}`, `M ≤ 200`). This run measures what
that recipe costs on the real INT8 artifact over a full Preset-A block, and verifies that all 448
tiles it produces are bit-exact — not just the 32 shapes the bisect probed.

> **`T_N`(exact recipe) = 0.7164 s/block** (9 GEMMs, 1536 tokens), **1.90× the 377.8 ms 9-graph
> stage** — at **140 GOPS** instead of 265 — plus **56 compilations / 8.6 s per block**, and
> **448/448 tiles bit-exact (Δ = 0 LSB)**.

Tools: `tools/pipe/tn_recipe.py` (host driver + verifier) ·
`proc_b --recipe 1 [--recipe-dump DIR]` (device, §"exact-recipe cost" in `proc_b.cpp`). Captures
`tn_recipe_20261116_013019`, `…_013210`, `…_013335`, `…_013625` (4 runs, all `block_00.bin`).

## Method

Per-tensor scales `SA = SW = 1/127`, zero bias, one graph per **column block** at `M = 192`, weight
operand a window into the **one** file-mapped INT8 artifact (`off + col0·K`) — no copies, still
route B. Activations are the `basis` pattern `A[r][r % K] = 127`, so the exact output element is one
stored weight byte (`out[r][n] = W[col0+n][r % K]`) and nothing saturates.

**The graph count is set by column blocks, not by rows × columns.** A graph is shape-defined; the
only per-block constant operand is the weight window, so one `M = 192` graph is re-executed over
each row tile's activation slice. A 1536-token block therefore needs

- **56 compilations** = 8 × 4 (the eight 1536-wide GEMMs) + 24 (`ffn.up`: 23 × 384 + tail 128)
- **448 executes** = 56 graphs × 8 row tiles

## Measured — the cost (4 runs, steady state = passes 2…5)

| | | 9-graph stage (P3.3, `RESULTS_proc_b_p33.md`) | exact recipe |
|---|---|---|---|
| `T_N` per block (1536 tokens, 9 GEMMs) | | **377.8 ms** (373.4–381.1, 3 runs) | **716.4 ms** (715.8 / 717.2 / 717.8 / 716.4 across 4 runs, **0.3 % spread**) |
| GOPS | | 265 | **140** |
| dispatch count | | 36 executes, 9 compilations | **448 executes, 56 compilations** |
| bit-exact output | measured Gate 3 | **25.8 %** | **100 %**, 448/448 tiles |
| first pass (cold weight fetch) | | — | 777.0 / 814.8 / 803.9 / 801.9 ms |

Per GEMM, median of passes 2…5 (all four runs agree to ≲0.5 %):

| GEMM | N_out | col blocks | execs/block | ms/block | GOPS | compile ms |
|---|---|---|---|---|---|---|
| `attn1.to_q` | 1536 | 4 | 32 | **51.03** | 142 | 665 |
| `attn1.to_k` | 1536 | 4 | 32 | 50.95 | 142 | 594 |
| `attn1.to_v` | 1536 | 4 | 32 | 52.20 | 139 | 594 |
| `attn1.to_out` | 1536 | 4 | 32 | 50.90 | 142 | 591 |
| `attn2.to_q` | 1536 | 4 | 32 | 51.54 | 141 | 591 |
| `attn2.to_k` | 1536 | 4 | 32 | 51.00 | 142 | 658 |
| `attn2.to_v` | 1536 | 4 | 32 | 51.41 | 141 | 594 |
| `attn2.to_out` | 1536 | 4 | 32 | 51.57 | 141 | 586 |
| `ffn.up` | 8960 | 24 | 192 | **306.16** | 138 | 3762 |

The **58 % throughput drop** (225 → 140 GOPS on the same GEMM, same weight bytes) is not one slow
shape: every column block lands at 139–142 GOPS, i.e. **the small exact shapes cost throughput
across the board** (`M = 192` is half a row tile of work per dispatch, and the 1.4–1.6 ms of
per-execute HAL overhead no longer amortises over 384 rows).

## Measured — the output is exact, tile by tile

With the `basis` activation the expected output is a stored weight byte, so the check is a byte
comparison against `block_00.bin` — no BLAS, no requant model to trust. Pass 1 of run 4 dumped all
448 tiles:

| GEMM | tiles | bit-exact | worst \|Δ\| (LSB) |
|---|---|---|---|
| the eight 1536-wide GEMMs | 32 each = 256 | **256/256** | 0 |
| `ffn.up` (incl. the tail 128 block at `col0 = 8832`) | 192 | **192/192** | 0 |
| **total** | **448** | **448/448** | **0** |

So the recipe holds **under repeated executes, over a whole block, on all nine GEMMs** — the
stronger form of P3.7b's per-shape claim. This is the first configuration in the project where the
NPU stage is both measured and correct.

## Consequence for the cut — reported, not applied

`solve_cut.py` with the measured pair (`T_G` 0.1474 co-run, `T_V` 10.64 hybrid):

| `T_N` fed to the solver | k_b\* | period | note |
|---|---|---|---|
| 0.6484 (locked, 9-graph co-run + downsep) | **160** | 23.61 s | the 25.8 %-exact stage |
| 0.9744 = 0.7164 exact + 0.2580 downsep | **166** | 24.47 s | **+3.6 % period for a correct stage** |
| 0.7164 exact, `ffn.down` not on the NPU | 162 | 23.88 s | if down runs on Mali |

The parity fix costs **0.86 s per chunk (3.6 %)** — the NPU stage's *share* of the block grows, so
the optimal cut moves later (more blocks to the GPU) and the period is set by `166 × 0.1474`.
`configs/rates_preset_a.json` is **not** edited and `k_b* = 160` stays locked: the 0.7164 is a
**solo** number and the locked 0.6484 is a co-run one, so the comparison is not yet like-for-like.

## The compilation count is a real cost, and it is amortisable

56 compilations cost **8.64 s** (154 ms each) and are **per block**, because the weight operand is
a window into that block's bytes. Twenty NPU blocks per chunk ⇒ **172 s of compile per chunk** if
recompiled every chunk — 46 % of the 370 s measured period, which would kill the recipe as a
pipeline proposition.

It is amortisable over the 10-chunk run, though: the **base** weight bytes for a given block are
the same in every chunk (only the detail-expert factors are step-dependent), so the 20 blocks'
graphs can be compiled once (≈172 s, one-time) and re-executed by later chunks. Not measured —
what is measured is that the compile exists, costs 8.64 s/block, and is separable from the exec.

## What this does not prove

- **The co-run rate.** 0.7164 s is a solo stage number; the 9-graph stage measured 377.8 solo vs
  390.4 co-run (1.03×), so the cut comparison above needs the same treatment before it is table-grade.
- **Block-to-block variation.** Only `block_00.bin`'s weights were used. Timing is shape-driven and
  the exec rate spread across four runs is 0.3 %, but the 100 % parity claim is one block's bytes.
- **`ffn.down`** (K = 8960) still has no graph (compiler K ceiling 1664), so it stays on Mali at
  47.33 ms or on the TPU as 7×K=1280 + 6×ADD at 258.0 ms — unchanged by this work.
- **Per-column scales.** The recipe is per-tensor, like every NNAPI graph on this HAL; the measured
  accuracy penalty stands at 2.15× mean GEMM error (`RESULTS_scales.md`).
- **Where the 19 % dispatch overhead goes.** Attributed to shape, not localized per-op; no mechanism
  claimed beyond the measured 139–142 GOPS at `M = 192` vs 225 at `M = 384`.

## Thermal

Runs were gated with `battery + 3 + 1.0 °C` slack (the TPU zone reports in whole degrees and the
idle floor sits between two steps — the case `tools/npu/cool_rerun.py` documents). Run 4 started at
**TPU 26.0 · skin1 24.0 · skin2 24.4 · VIRTUAL-SKIN 23.8 · battery 22.5** (gate line 25.5, i.e.
**0.5 °C over** at the floor, floor-grade and stated) and the 12.6 s run moved the TPU zone
26.0 → 33.0 °C. No drift term applies inside a 0.72 s stage.

---

## P3.7e — the pass-major binary is exact *and* 9.7 % cheaper (capture `tn_recipe_20261116_023152`)

P3.7c's 448/448 parity dump was taken with the **gemm-major** `run_recipe` (each GEMM freed its
slots before the next compiled). P3.7d rewrote it **pass-major** — all 56 graphs compiled up front,
then each pass walks the block — so a pass is contiguous in time and can be classified solo vs
contended. Parity under that binary had not been re-checked. It has now.

| quantity | gemm-major (P3.7c) | **pass-major (P3.7e)** |
|---|---|---|
| block, steady (passes 2…5) | 716.4 ms | **646.69 ms** median (646.4 / 646.7 / 646.7 / 647.0, stdev **0.21 ms = 0.03 %**) |
| block, pass 1 (cold) | 777–815 ms | 668.5 ms |
| ratio vs the 377.8 ms 9-graph stage | 1.90× | **1.71×** |
| GOPS over 50.13 GMAC | 140 | **155** |
| compilations (wall) | 56 (8.64 s) | 56 (**9.63 s**) |
| dumped tiles bit-exact | 448/448 (Δ = 0) | **448/448 (Δ = 0 LSB, worst 0)** |

So the pass-major rewrite is not only a *classification* device: holding all 56 graphs at once
removes the per-GEMM compile gaps that P3.7c paid **inside** every pass, and buys **9.7 %** off the
block. Compile grows 8.64 → 9.63 s (all 56 resident, matching P3.7d's 9.57 s).

Gate: TPU **27.0** · skin1 24.6 · skin2 25.2 · VIRTUAL-SKIN 24.4 · battery 23.2 → gate **27.2**,
**PASS at start** (whole-degree TPU sensor, 1.0 °C slack as `cool_rerun.py` documents). After the
13.2 s run: TPU **34.0**.

### The number that does not fit — reported, not resolved

P3.7d priced contention as a non-event by comparing **700.5 ms contended** against **707.0 ms** in
"the passes before the GPU joined". Tonight's clean solo pass-major run is **646.7 ms** — i.e. both
of P3.7d's numbers sit ~9 % *above* a floor-started solo run, and they bracket it from either side.
Consequences, stated as candidates with the measurement that would separate them:

- **(i) thermal.** The co_run's later passes ran with the TPU zone climbing to **45 °C** (it ended
  there), while this run spanned 27 → 34 °C. The 9-GEMM soak showed **+0.35 %** drift over 25 → 31 °C,
  so a derate above ~34 °C would be **new** for the recipe — plausible (56 compiles + 448 executes
  per block touch the HAL far more often) but not measured.
- **(ii) order/state.** The co_run had just finished the GPU's 8 solo blocks when pass 1 started
  (CL allocations live, 46 MB of weight pages competing). The first four passes there
  (719.95 / 708.35 / 709.63 / 710.96) never fell to 647 even though the device was *cooler*
  at start (TPU 26.0) than tonight's — which thermal does not explain on its own.

**What this run does not do is claim contention is 8 %.** It removes the evidence for "contention is
a non-event" from the table and names the run that settles it: a **floor-started co_recipe** whose
crossing passes can be compared to 646.7 ms rather than to a 707 ms baseline taken above the floor.

### Cut, re-solved on both candidates (reported, not applied — lock untouched)

`T_N` = recipe + `ffn.down` downsep 0.2580 s (**still solo**):

| `T_N` candidate | `T_N` | `k_b*` | period | vs locked 160 / 23.61 s |
|---|---|---|---|---|
| 0.6467 solo (this run) | **0.9047 s** | **165** | **24.32 s** | reported |
| 0.7005 contended (P3.7d) | 0.9591 s | 166 | 24.47 s | reported |

The cut is insensitive to which of the two you believe (**165 vs 166**, 0.6 % in period) — the
lock-breaking question is not this number.

### Two host-side defects fixed en route (they silently blocked the run)

1. **`run(["bash", …])` resolved to WSL bash, not git-bash.** `subprocess` searches `System32`
   before `PATH`, so the build never ran: `/bin/bash: C:/…/build.sh: No such file or directory`
   (rc 127, `--build` returned 2 before touching the phone). WSL mounts the repo at `/mnt/c`, so
   neither `C:/…` nor `/c/…` resolves. Fixed in `tools/probe/probe_host.py::run()` — a `bash`/`sh`
   argv[0] is now resolved through `bash_bin()` (git-bash, `GIT_BASH` overridable) with forward-slash
   arguments. Every tool in the repo inherits the fix.
2. **A single confirm sample aborted a floor-grade run.** TPU reads 27.0 (PASS, gate 27.4), then
   28.4 on the 8 s confirm at an unchanged battery → `NO TIMED RUN`. `gate_wait` now returns to the
   wait loop on a failed confirm instead of failing the gate (still bounded by `--max-wait`); the
   device passed 53 s later at 27.0 with no cooling applied.
