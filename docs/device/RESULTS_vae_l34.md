# P3.16 — layer 34 is neither its residual add's grid nor the `conv_out` shape

*2026-11-16 09:30. Host-only, **no device run and no device rate touched**: `T_V`, `T_N`,
`T_G` and `k_b*` are exactly as committed. Instrument `tools/npu/vae_sim.py` (extended:
`--layer34` → arms `add34` / `coutpc` / `both34`), capture
`docs/device/vae_sim_l34_20261116_092942.{json,md}`, latent
`artifacts/vae/dump_20261116_080721` (the post-fix device walk's own latent).*

## What was unrun

P3.15 closed the accumulating scheme (bias over-count, per-tap rounding) at ±0.3 dB and
named **layer 34** — `decoder.up_blocks.3.resnets.2.conv2`, cosine **0.7149**, the worst
layer in all five arms — as the next defect site, with two candidates it could not price:

1. the **residual add** after conv2, written on the calibrated `post_add_sa` grid;
2. the **`conv_out` 96→3 shape**, three channels on one output scale.

Both are arms, not opinions, so both were measured.

## Method — four arms, one instrument

`vae_sim.py --layer34` walks `vae_wire.txt` with the measured `shift` gather and the
device's own semantics (per-tap bias, per-tap rounding), changing one thing at a time:

| arm | change |
|---|---|
| `wire` | nothing — the program as `npu_block.cpp` builds it |
| `add34` | the residual add after conv 34 summed in **float32**, no int8 hop; the next op (`norm_out`, an L2 channel norm) does the only rounding |
| `coutpc` | `conv_out` given **three per-output-channel** output scales (`max/127` from the reference module's own per-channel max) at both requant and dequant |
| `both34` | both |

**Validator first:** the `wire` arm reproduces P3.15's capture to the digit — **20.5876 dB
/ cosine 0.888501 / worst layer 34 0.714915** — so the arms below are comparable to the
earlier work and to the phone (P3.15 measured the simulator tracking the device to
**0.15 dB**: 20.59 vs 20.44).

## Result 1 — the add's grid is worth 0.0016 dB

| arm | final PSNR (2.0) | cosine | Δ PSNR vs `wire` | Δ cosine |
|---|---|---|---|---|
| **`wire`** | **20.5876 dB** | 0.888501 | — | — |
| `add34` | 20.5892 dB | 0.888549 | **+0.0016** | +0.000048 |
| `coutpc` | 20.5891 dB | 0.888635 | **+0.0015** | +0.000134 |
| `both34` | 20.5907 dB | 0.888681 | **+0.0031** | +0.000180 |

**Both named candidates together are worth 0.003 dB** against a decode that is ~18 dB
short. Layer 34 stays at **0.7149** in all four arms — the worst layer does not move at
all, which is the cleanest possible statement that neither candidate is what produces it.

## Result 2 — the grid question, measured at all 14 residual adds

Every add writes its sum on one calibrated grid (`post_add_sa`, from the post-add max).
`ratio` = that grid's LSB over the **finer** operand's own LSB, i.e. how much the branch's
quantisation step is multiplied by at the add:

| add after conv | sa(act) | sa(residual) | out sa | ratio | clamp |
|---|---|---|---|---|---|
| 2 | 0.074764 | 0.012566 | 0.07705 | **6.13×** | 0.050 % |
| 4 | 0.028660 | 0.07705 | 0.066555 | 2.32× | 0.040 % |
| 6 | 0.026496 | 0.066555 | 0.066452 | 2.51× | 0.046 % |
| 8 | 0.036527 | 0.066452 | 0.066341 | 1.82× | 0.097 % |
| 10 | 0.030339 | 0.066341 | 0.060082 | 1.98× | **0.520 %** |
| 15 | 0.042721 | 0.069312 | 0.084059 | 1.97× | 0.068 % |
| 17 | 0.060881 | 0.084059 | 0.084683 | 1.39× | 0.214 % |
| 19 | 0.062944 | 0.084683 | 0.11059 | 1.76× | 0.008 % |
| 23 | 0.076883 | 0.15048 | 0.15783 | 2.05× | 0.008 % |
| 25 | 0.079617 | 0.15783 | 0.14866 | 1.87× | 0.026 % |
| 27 | 0.099491 | 0.14866 | 0.21329 | 2.14× | 0.005 % |
| 30 | 0.063959 | 0.23980 | 0.26677 | 4.17× | 0.009 % |
| 32 | 0.060177 | 0.26677 | 0.28620 | **4.76×** | 0.001 % |
| **34** | 0.080382 | 0.28620 | 0.33601 | **4.18×** | 0.002 % |

Two corrections to the record, both measured:

- **P3.11's "post-add range 4.76× conv2's own" is the add after conv 32, not layer 34.**
  Layer 34's own ratio is **4.18×** (0.33601 / 0.080382). The full spread across the 14
  sites is **1.39× – 6.13×**.
- The add's clamping is negligible everywhere (**worst 0.52 %**, after conv 10), so the
  coarsening never costs information by saturation — it costs only LSB noise, and Result 1
  prices that at 0.002 dB.

## Result 3 — `conv_out`'s three channels are within 1.15× of each other

The reference module's own per-channel maxima are **1.1442 / 1.0510 / 1.2086** — a **1.15×
spread**, not the 2–4× a per-channel scheme would need to pay for itself. The artifact's
single `sa_out` is `0.0112168`, calibrated from a max of 1.4245, which is **18 % above the
largest channel max this walk produces** (1.2086) — the calibration is conservative, not
starved.

Per-channel scales are therefore worth **+0.0015 dB**, and there is a trap inside that
number worth recording: the `coutpc` walk scored **20.4792 dB / 0.885514 if its codes are
dequantised at one scale** and **20.5891 / 0.888635 only when each channel is dequantised
at its own**. Per-channel requant without per-channel dequant is **−0.11 dB**, i.e. worse
than doing nothing. The gain and the cost are the same size, opposite in sign.

## What this changes

- **Layer 34 is not the add and not the 96→3 shape.** Both candidates are refuted at the
  end-product level (0.003 dB combined, worst layer unchanged). P3.15's ranking is paid
  off, and the last two cheap static-scheme candidates in the decode are gone.
- What is left at layer 34 is a distinction this instrument cannot yet draw: its input
  (conv1's output) is already at **0.8618**, and a 3×3×3 conv either amplifies that
  inherited error or adds its own. The discriminating probe is an **oracle-input arm** —
  feed each conv the reference's own input quantised at the program's `SA_in` and score
  its output. Intrinsic-bad (~0.85) ⇒ the conv/weights/scale at that site;
  input-inherited (~0.99) ⇒ the error is upstream and layer 34 is only where it lands.
  That is host-only and is the named next action.
- With the static schemes closed and the pqc branch waiting on a decision, the decode's
  remaining ~18 dB is **accumulation**, not a bug: the per-layer cosine drifts
  0.996 → 0.94 → 0.83 through the up_blocks with no single step, which is what a 36-layer
  int8 chain at 20 dB looks like.

## Not claimed

No device number, so **no rate is re-derived**: `T_V` = 8.908 s, `T_N` exact 0.8905 s/block,
`k_b*` = 165 all stand. No mechanism for the layer-34 drop. **`post_quant_conv` is still
unapplied** — it remains the only measured item that needs the student's call (+7.32 dB,
P3.15), and it edits the artifact. Nothing locked was touched.

## Method notes

- `add_fp` is keyed by the **conv index before the add**, so an arm changes exactly one
  site; `cout_pc` keys on the layer's `ai` and needs `chan_sa` populated from the
  reference trace (which now records per-layer `chmax` for every layer, not just `conv_out`).
- Two bugs were paid for by the run itself and never reached a number: `chan_sa` was
  referenced before its declaration in the arm matrix, and the per-channel dequant
  broadcast on dim 0 instead of dim 1 (the activation is `[frames, channels, h, w]`).
- Runtime: 1 fp32 reference (44.4 s) + 4 walks (3 min 35 s each) = **14 min 50 s**, one
  process. A walk is 36 convs of up to 113 M elements (layer 34 is `[9, 96, 272, 480]`).
