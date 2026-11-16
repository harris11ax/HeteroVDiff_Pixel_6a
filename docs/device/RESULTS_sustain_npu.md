# Sustained NPU run — does the Edge TPU hold `T_N` for a whole chunk?

*Measured 2026-11-15 21:41:49, Pixel 6a `25311JEGR11811`. Capture:
`docs/device/sustain_npu_20261115_214149.json` (tool: `tools/npu/sustain_npu.py`).*

## The question

Every `T_N` number in the cut so far comes from a **short burst**: 9 GEMMs compiled
once, then 100 rounds of `tiles=4` executes (`RESULTS_npu_tn.md`,
`RESULTS_corun_tg.md`). The cut then assumes that rate survives the NPU stage of a
real chunk — **20 back-to-back blocks (12.97 s) + the 10.64 s hybrid decode ≈ 23.6 s
of continuous Tensor-G1 work inside a 23.61 s period**. If the TPU derates at second
15 of every chunk, the period rises and no burst capture would ever show it.

This run drives the existing `npu_block corun` loop (9 compilations held live, GPU
idle) long enough to cover the whole continuous length, and reports the drift with
the thermal state logged before and after.

## Method

`python tools/npu/sustain_npu.py --rounds 60` → `npu_block corun 4 60 384 0`
(per-tensor scales, `PREFER_SUSTAINED_SPEED`, M=384 = the measured NNAPI ceiling,
`tiles=4`, `iters`/rounds = 60). One round = one block's 9 compilable GEMMs.
60 rounds × 0.410 s = **24.6 s of continuous TPU execution**, i.e. slightly longer
than the NPU stage it is standing in for.

Verdict threshold is **2 %** on the last-20 % / first-20 % round means — half the
±4 % per-GEMM spread already measured under co-run, so a pass cannot be noise.

## Thermal

| | TPU | VIRTUAL-SKIN | skin_therm1 | skin_therm2 | battery |
|---|---|---|---|---|---|
| before | 25.0 | 22.9 | 23.1 | 23.5 | 22.0 |
| after  | 31.0 | 23.7 | 24.4 | 25.3 | 21.9 |

Gate: `max(TPU, skin, VIRTUAL-SKIN) = 25.0 °C` vs `battery + 3 = 25.0 °C` — the run
started **exactly on** the line (table-grade), and the TPU zone rose **6.0 °C**
during it without the per-round cost moving. That is the interesting part: 6 °C of
soak is not enough to trigger a derate here.

## Result

| | ms per round |
|---|---|
| first 20 % (rounds 0–11) | 408.902 |
| last 20 % (rounds 48–59) | 410.346 |
| all 60 rounds: mean | 410.225 |
| stdev | 4.787 |
| min / median / max | 386.936 / 410.271 / 419.059 |

**Drift = +0.35 %** over 24.6 s (least-squares slope **+0.0299 ms/round**, i.e.
+1.8 ms over the whole run). Both are inside the noise floor: **the rate is
sustained.**

Per-GEMM drift, first 5 % vs last 5 % of each GEMM's own executes (240 samples each):

| GEMM | drift | | GEMM | drift |
|---|---|---|---|---|
| `attn2.to_v` | +2.2 % | | `attn1.to_v` | −0.3 % |
| `attn1.to_q` | +1.8 % | | `attn2.to_out` | +0.3 % |
| `attn1.to_k` | −0.7 % | | `ffn.up` | +0.2 % |
| `attn2.to_q` | −0.7 % | | `attn2.to_k` | −0.1 % |
| `attn1.to_out` | +0.3 % | | | |

No GEMM shows a systematic end-of-run penalty, and `ffn.up` — the single largest op
(35.3 ms/tile, 36 % of the round) — moves by +0.2 %. Harness summary for the run:
**0.4037 s/block, 248.4 GOPS** (solo; the co-run table's 9-GEMM part is 0.3904 s).

## What this buys the cut

- **The 23.5 s "throttling" open item is closed for the FC class.** `T_N` = 0.6484 s
  needs no thermal derating term over a chunk: the 9-GEMM part holds to +0.35 %
  across 24.6 s, which is longer than the stage.
- The 2 % threshold was not nearly approached, so this is not a "just passed" result
  — it is a null result with 6 °C of headroom consumed.
- Side observation: the sustained solo rate (0.4037 s/block, 248.4 GOPS) sits between
  the solo table's 0.4178 and the co-run 0.3904. The co-run figure being the *fastest*
  is consistent with what `RESULTS_corun_tg.md` already flagged (the GPU also got
  faster under load); this run does not explain it, and does not need to.

## Still open (stated, not modelled)

1. **`ffn.down` under contention** — its 0.2580 s is a solo number (`downsep`, 7×FC +
   6×ADD). `corun` holds only the 9 block GEMMs, so there is no co-run figure for it.
   Stage total `T_N` = 0.6484 s therefore mixes a co-run part and a solo part.
2. **The decode's `CONV_2D` class** (10.554 s, the other half of the stage) was not
   soaked for 23.5 s — it is a different op class on the same TPU, measured
   gate-compliant but not continuously.
