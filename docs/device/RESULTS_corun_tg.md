# Co-run `T_G` / `T_N` — both engines at once, and the measured cut

*Run 2026-11-15 19:48:09, capture `corun_tg_20261115_194809.{json,md}` (JSON gitignored, the
`.md` sibling is committed). Harness: `tools/corun/co_run.py` + `tools/corun/corun_device.sh`,
with `TS|gpu|…` / `TS|npu|…` absolute `CLOCK_MONOTONIC` stamps added to
`tools/android/gemm/gemm_bench.cpp` and `tools/android/npublock/npu_block.cpp`.*

Every rate in the cut so far was **solo**: one engine at a time. The cut is only allowed to use
stage rates, and the paper's model assumes the two stages run concurrently — so the question this
run answers is whether the Mali-G78 and the Edge TPU actually hold their solo rates while both are
executing, given that both read the same file-mapped INT8 artifact.

## Method (and why it is an A/B, not a cross-capture contrast)

Two processes on the phone, one `adb shell`:

1. `gemm_bench block` — a whole Preset-A block (10 GEMMs, M = 1530 tokens, variant
   `tiled_64x64x32_t8x4`), 10 iterations **solo** in this session.
2. `npu_block corun` — **new mode**: compiles the 9 compilable block GEMMs once, holds all nine
   compilations, then loops 100 rounds of 4 executes per GEMM (one round = a block's worth of NPU
   work minus `ffn.down`). Compiles are deliberately *not* interleaved with the timed work; in
   `single` mode they would contend with the GPU process on the CPU, which is not a stage rate.
3. The NPU process starts **first** and runs ~1.5 s alone, so genuine solo samples and contended
   samples come from the **same process, same code path, same thermal state**.
4. The GPU joins and runs 25 block iterations. Overlap is derived from the timestamps, not
   asserted: **99.8 % of GPU-time fell inside NPU executes**.

`ffn.down` is excluded from the NPU load because it has no compilable graph at any K > 1664
(`RESULTS_npu_tn.md` §4); its chunked cost is added separately below and flagged.

## Thermal

Start **TPU 26.0 · skin_therm1 24.3 · skin_therm2 24.9 · VIRTUAL-SKIN 24.0 · battery 22.6**; the
locked gate is `max(TPU, skin, VIRTUAL-SKIN) ≤ battery + 3 = 25.6 °C`, so the start is **0.4 °C
above the battery-driven gate** — the known artefact of a TPU zone that reports in whole degrees and
sits at its idle floor (26.0 for 20 consecutive samples over the preceding 5 min). It is the same
floor the table-grade `T_N` and `T_V` captures sat on. End of run: **TPU 40.0 · skin_therm1 30.9 ·
skin_therm2 31.1 · VIRTUAL-SKIN 28.4 · battery 22.9** — a 14 °C rise in a 12 s co-run, i.e. both
engines really were executing.

## GPU — Preset-A block, solo vs co-run (same session)

| LoRA mode | condition | n | min ms | median ms | mean ms |
|---|---|---|---|---|---|
| 0 (base block) | solo | 10 | 156.28 | 160.02 | 160.00 |
| 0 (base block) | **co-run, NPU executing** | 25 | **146.02** | **147.41** | **147.38** |
| 1 (LoRA add, FP16) | solo | 10 | 159.67 | 162.86 | 162.81 |
| 1 (LoRA add, FP16) | co-run | 25 | 148.70 | 150.33 | 150.33 |
| 2 (rank-128 FP16 factors) | solo | 10 | 643.88 | 646.71 | 647.62 |
| 2 (rank-128 FP16 factors) | co-run | 25 | 613.16 | 617.29 | 617.98 |

**The GPU does not slow down under a full NPU load — it measured 8 % faster** (147.4 vs 160.0 ms
median). Per-iteration solo samples span 156.3–163.9 ms with the first one highest (cold weight
fetch); the co-run samples are a tight 146.0–147.9 ms **from the very first iteration**, so this is a
device-state difference between the two processes, not a within-run clock ramp. The candidate
explanation is SoC frequency governance — the NPU driver threads keep the cluster busy, so the GPU
process never runs from an idle DVFS state — but **that is not established by this measurement and
is not claimed**; what is established is that contention is below the noise floor of the run.

Consequence for the cut: `T_G` (co-run) = **0.1474 s/block** for the base Preset-A block. The
previously committed solo value (`0.1618 s`) stays the conservative figure; the pair below is
solved both ways.

## NPU — per GEMM, solo vs contended (identical process)

| GEMM | n solo | solo mean ms | n contended | contended mean ms | Δ |
|---|---|---|---|---|---|
| `attn1.to_q` | 168 | 7.987 | 232 | 7.991 | +0.0 % |
| `attn1.to_k` | 161 | 7.754 | 239 | 7.807 | +0.7 % |
| `attn1.to_v` | 160 | 7.602 | 240 | 7.762 | +2.1 % |
| `attn1.to_out` | 160 | 7.616 | 240 | 7.637 | +0.3 % |
| `attn2.to_q` | 161 | 7.650 | 239 | 7.553 | −1.3 % |
| `attn2.to_k` | 164 | 7.594 | 236 | 7.611 | +0.2 % |
| `attn2.to_v` | 167 | 7.939 | 233 | 7.613 | −4.1 % |
| `attn2.to_out` | 168 | 7.636 | 232 | 7.607 | −0.4 % |
| `ffn.up` | 168 | 35.816 | 232 | 35.709 | −0.3 % |

Round time (9 GEMMs × 4 tiles): **median 390.4 ms, min 366.0, max 435.2** over 100 rounds —
**0.3904 s/block**, 257.3 GOPS, against the gate-compliant solo table's 0.4178 s/block (222 GOPS).
Per-GEMM deltas are inside ±4 %, i.e. the TPU is **latency/dispatch-bound, not bandwidth-bound**:
a 384-token `FULLY_CONNECTED` costs 7.6–8.0 ms whether or not a 1.77-TOPS GPU is hammering DRAM
next to it.

## The stage rates this run licenses

| quantity | value | provenance |
|---|---|---|
| `T_G` co-run | **0.1474 s/block** | median of 25 base-block samples, 99.8 % overlapped |
| `T_N` co-run, 9 GEMMs | **0.3904 s/block** | median of 100 rounds in the same run |
| `T_N` co-run, `ffn.down` | **0.2580 s/block** | `…_174838` `downsep`, **solo** — not re-measured under contention (no compilable single graph, so it cannot be driven by `corun`) |
| **`T_N` co-run (sum)** | **0.6484 s/block** | 0.3904 + 0.2580, a measured sum |
| `T_V` | **10.554 s/chunk** (TPU) | `RESULTS_vae_tv.md` §3, still a lower bound (mid-block attention unmeasured) |

`T_N / T_G` = **4.40×** co-run (4.18× on the solo pair) — the ordering does not depend on the
contention question.

## The measured cut

`python phase2_profile/solve_cut.py` (selftest PASS; paper anchor reproduces `k_b = 51`):

| rates | `T_G` | `T_N` | `T_V` | `k*` | `k_b*` | period | vs NPU-only |
|---|---|---|---|---|---|---|---|
| **co-run pair, TPU decode** | 0.1474 | 0.6484 | 10.554 | 159.92 | **160** | **23.58 s** | **5.40×** |
| solo pair (previously committed) | 0.1618 | 0.6758 | 10.554 | 157.83 | 158 | 25.56 s | 5.17× |
| co-run pair, Mali decode | 0.1474 | 0.6484 | 12.256 | 162.06 | 162 | 23.93 s | 5.39× |

**Committed: `k_b* = 160`** for the TPU-decode assignment (`T_G = 0.1618` conservative bound
⇒ 158; the two agree to ±1.3 % and the period differs by 2.0 s). The GPU takes blocks `[0,160)`,
the NPU takes `[160,180)` and then the decode; the cut is interior and nearly balanced
(GPU stage 23.58 s, NPU stage 23.52 s, idle 0.06 s).

Two consequences worth carrying into the paper:

- **The cut is sharp.** Step-level granularity (`k_b` a multiple of `L = 30`) costs 27 %
  (k_b = 150 → 30.0 s). A 15-block granularity would already be needed to keep regret under ~3 %.
- **`T_V` dominates the NPU stage**: 10.554 of its 23.52 s is decode. Ignoring `T_V` in the solve
  moves the cut to `k_b = 147` and costs 35 % — the decode is exactly the term the paper's
  model needs and the one this port must get right.

## What this does **not** establish

- Contention was measured with the NPU driven by `FULLY_CONNECTED` graphs and the GPU by the tiled
  GEMM. A Phase-3 pipeline where both stream the *same* artifact pages may behave differently;
  the harness here maps the weights but does not interleave page faults by construction.
- The co-run covers 100 rounds ≈ 10.4 s of NPU work. A 90-block NPU stage (~58 s) is long enough
  for sustained throttling to matter, and that is a different question from contention.
- `ffn.down` is not in the contended number (see above), and mid-block VAE attention is still
  unmeasured, so `T_V` remains a lower bound.
- The weights are the deterministic LCG fill: these are shape-determined latencies, not accuracy
  results (`phase1_quant/` still has no quantiser).
