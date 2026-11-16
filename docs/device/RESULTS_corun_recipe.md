# P3.7d — the co-run treatment of the exact NPU recipe

*Measured 2026-11-16 02:09 on `25311JEGR11811`, capture `corun_recipe_20261116_020926`
(pilot `…_020535`, pass-major sanity `tn_recipe_20261116_020353`). Tool:
`tools/corun/co_recipe.py` + `tools/corun/co_recipe_device.sh`; device side
`proc_b --recipe 1` (pass-major) and `gemm_bench block`, two processes, one device shell.*

**Why this run exists.** P3.7c priced the only bit-exact NPU geometry at **0.7164 s/block**, but
that number is **solo**, while the locked `T_N` (0.6484 s) and `T_G` (0.1474 s) are **co-run**
numbers. A cut argument may not mix the two. This run times both engines together and answers one
question: what does the exact recipe cost **while the GPU is executing a Preset-A block**.

> **⚠ Superseded on one point — read `RESULTS_tn_recipe.md` §P3.7e before quoting the −0.9 %.**
> A floor-started solo run of the same pass-major binary measures **646.7 ms/block** (2026-11-16
> 02:31, capture `tn_recipe_20261116_023152`, TPU 27.0 °C, stdev 0.03 %). This run's 707.0 "solo"
> and 700.5 "contended" therefore both sit **~9 % above** a floor-started solo block and bracket it,
> so the "contention is not measurable" reading below is **not established**. Untested candidates:
> a TPU derate above ~34 °C (this run ended at TPU 45 °C) or order/state (the four pre-GPU passes
> read 708–720 at a *cooler* 26.0 °C start). The pass-major rewrite, the overlap proof, the GPU
> `146.05` vs `160.27` ms pair and the exactness chain are unaffected. What settles it: a
> **floor-started** `co_recipe` run whose crossing passes are compared to 646.7 ms.

## The one code change that made the measurement possible

`run_recipe` used to be **gemm-major** (for each GEMM: all passes). A "pass" then spanned the whole
run, so no pass could be classified as "ran alone" vs "ran while the GPU was busy". It is now
**pass-major**: all 56 column graphs are compiled up front, then each pass walks the block
GEMM by GEMM. That also makes each pass contiguous in time, and every `RECIPE|pass` /
`RECIPE|col` line carries `t0`/`t1` in **CLOCK_MONOTONIC ms** (same clock `gemm_bench` stamps
its `TS|gpu` samples with), so overlap is computed from the artifact instead of assumed.

Cost of the change: all 56 graphs are held at once (P3.7c freed each GEMM's slots before the next
GEMM compiled). It works — 448 executes/pass, no compile failure, PSS unchanged in kind from P3.3.

## Numbers

| quantity | value | source |
|---|---|---|
| exact recipe, **under GPU load** (sum of per-GEMM medians, contended passes) | **700.5 ms/block** | `…_020926` |
| exact recipe, **before the GPU joined** (same run, passes 2–4) | 707.0 ms/block | `…_020926` |
| exact recipe, device `SUMMARY` block (all 24 passes) | 701.1 ms median, pass 1 720.0 ms | `…_020926` |
| exact recipe, pilot capture (8 s GPU window) | 698.4 ms/block contended | `…_020535` |
| exact recipe, solo, pass-major sanity (3 passes) | 710.3 ms/block (721.5 pass 1) | `tn_recipe_…_020353` |
| exact recipe, solo, P3.7c reference (gemm-major) | 716.4 ms/block | `RESULTS_tn_recipe.md` |
| GPU Preset-A block **while the recipe executes** | **146.05 ms** (n=25, min 145.6) | `…_020926` |
| GPU Preset-A block solo, same session | 160.27 ms (n=8) | `…_020926` |
| engine-time overlap (GPU time inside NPU executes) | **100.0 %** | computed from t0/t1 |
| compilations / executes per block | 56 / 448 | `…_020926` |
| compile wall per block (56 graphs) | **9 572 ms** | `…_020926` |
| per-GEMM contention penalty | **−1.3 … −0.4 %** (i.e. inside noise) | `…_020926` |

**The finding: contention is not measurable — the exact recipe costs the same whether the GPU is
hammering or idle (700.5 vs 707.0 ms, −0.9 %).** That is consistent with the 9-GEMM stage
(`RESULTS_corun_tg.md`: 390.4 ms co-run vs 417.8 ms solo) and with the sustained soak
(`+0.35 %` drift over 24.6 s): on Tensor G1 the two engines are not fighting over a shared
bandwidth budget at these rates.

The GPU, meanwhile, measured **9.2 % faster during the co-run than in the solo phase of the same
run** (146.05 vs 160.27 ms). The locked capture showed the same direction (0.1474 co-run vs
0.1618 solo, 8 %), so the effect reproduces, **but its cause is still DVFS, not something this
run established** — the solo phase always runs first and cooler, so load and thermal state are
confounded. `T_G` for the cut therefore stays the locked co-run **0.1474 s**.

## Cut, re-solved (T_V = 10.640 s hybrid, `solve_cut.py`)

| pair | T_G | T_N | k_b* | period |
|---|---|---|---|---|
| **this run, contended exact recipe + downsep** | 0.1461 | **0.9585** | **166** | **24.25 s** |
| P3.7c solo recipe + locked T_G | 0.1474 | 0.9744 | 166 | 24.47 s |
| locked (9-graph stage, `configs/rates_preset_a.json`) | 0.1474 | 0.6484 | **160** | 23.61 s |

`T_N` here = 0.7005 s measured under load + **0.2580 s `ffn.down` downsep, which is still a SOLO
number** (the downsep split has no co-run measurement yet). **Reported, not applied**: a
bit-exact NPU stage costs **+2.7 % period at `k_b* = 166`**, and the choice between "exact stage,
slower" and "9-graph stage, 25.8 % of tiles correct" is the student's.

## What this proves / does not

- proves: the exact recipe's per-block cost **with the GPU executing**, from a gate-compliant
  start (TPU 26.0 °C, gate 26.8 °C), with overlap proven from two processes' CLOCK_MONOTONIC
  stamps; and that contention adds ≤1 % to either engine at these rates.
- does not prove: **parity under the new pass-major order** (P3.7c's 448/448 bit-exact dump was
  taken with the gemm-major binary — same graphs and executes, but re-run `tn_recipe.py --dump`
  before quoting exactness again); `ffn.down` under contention; whether the 56 graphs survive
  across chunk boundaries, which is the only way the 9.6 s/block compile amortises; and
  block-to-block variation (one block region).
- the pilot `…_020535` shows why the control matters: with only passes 2–3 as the "solo" side its
  in-run control read 653.8 ms (a 2-sample median), which would have implied a false 7 %
  contention penalty. The later capture, with a 3-pass control and 24 passes, reads −0.9 %.

---

## P3.7g — floor-started co-run, 99.1 % overlap, TPU 26 → 45 °C: no contention term **and** no derate

Capture `corun_recipe_20261116_024049`, `--delay 6 --reps 24 --gpu-iters 60`, gate **PASS at start**
(TPU 26.0 · skin1 24.2 · skin2 24.8 · VIRTUAL-SKIN 24.0 · battery 22.7).

| quantity | this run | reference |
|---|---|---|
| exact recipe, **contended** (per-pass total) | **635.8 ms/block** | 700.5 ms in `…_020926` |
| per-pass warm median (n = 24) | **637.3 ms** (634.5 – 659.4; pass 10 = 659.4 is the GPU's first block) | 701.1 ms |
| GPU base block, solo → contended | **158.95 → 146.08 ms** (n = 8 → 43) | 160.27 → 146.05 ms |
| overlap (CLOCK_MONOTONIC) | **99.06 %** of GPU-time inside recipe executes | 100.0 % |
| TPU zone over the run | **26.0 → 45.0 °C** | 26.0 → 45.0 °C |

**Read the delay correctly: it starts at launch, and pass 1 comes after the ~9.6 s compile.** So the
GPU had already joined when pass 1 ran — `analysis.npu_solo_*` is `null` for exactly that reason, and
**every pass here is contended**. That is what makes the result sharp:

- **Contended is not slower.** 635.8 ms contended (this run) against **646.7 / 647.9 ms solo** (two
  GPU-free runs, P3.7e/P3.7f) — the recipe is, if anything, marginally *faster* with the GPU busy,
  the same direction the GPU itself shows (146 vs 159 ms). **P3.7d's 707.0 ms "solo" baseline is not
  reproducible and its −0.9 % "contention is a non-event" should be replaced by this.**
- **No thermal derate at 45 °C.** The nine contended passes after the join read 633.6–639.0 ms while
  the TPU zone climbed to 45 °C — flat, consistent with P3.7f's 40-pass series (+0.16 % over 26→33 °C).
  So "derate above ~34 °C" is **dead** as an explanation.

**Which leaves the 707 ms of `…_020926` unexplained by either candidate, and it matters.** The 02:09
run and tonight's runs use the *same* binary and the *same* block. The measured difference is 11 %.
The one environmental variable known to have changed between them: **the device's network/background
services were ON at 02:09 and OFF for every run since** (switched off at 02:2x to reach the idle
floor). A background-loaded CPU inflating per-execute wall time (448 executes/block, each a
synchronous NNAPI round trip) is the leading, *untested* candidate. If it holds, every rate measured
with the network on carries a ~10 % penalty, which would reach back into `RESULTS_stream_p35.md`
(32.72 s period) — so it is worth one cheap A/B before anyone quotes those as clean.

**Next measurement (host-cheap, ~25 s of device time):** at the floor with **network ON**, run
`python tools/pipe/tn_recipe.py --reps 8`. Floor + network on + ~707 ms ⇒ background load is the
cause; floor + network on + ~647 ms ⇒ the 02:09 numbers are a device-state artifact of that session.
