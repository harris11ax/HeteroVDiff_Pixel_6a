# Gate 1 — end-to-end PSNR/SSIM: FP32 vs the INT8 artifact's weights

`20261116_054851` · latent `[1, 16, 3, 34, 60]` seed 0 · configs 1–2 are **weight-only**
(quantise-dequantise); config 3 adds the **INT8 activation** operand at the per-layer `SA`
`calibrate.py` measured, i.e. the first end-to-end number that has both operands quantised.

| config | layers quantised | PSNR | SSIM | verdict |
|---|---|---|---|---|
| all quantised | 36/36 | **40.22 dB** | **0.9708** | **PASS** |
| quantised except worst 2 by weight cosine | 34/36 | **48.99 dB** | **0.9962** | **PASS** |
| all quantised + INT8 activations (measured SA) | 36/36 | **38.73 dB** | **0.9672** | **PASS** |

Gates: PSNR ≥ 28.0 dB, SSIM ≥ 0.92.

Round-trip: every rebuilt weight reproduces the artifact manifest's recorded per-layer relFro (max |diff| 4.5e-07), so the OHWI tap inversion is the artifact and not a plausible-looking transpose.

Per-frame PSNR (all-quantised): f0 53.49, f1 38.47, f2 43.21, f3 41.85, f4 40.89, f5 37.88, f6 39.33, f7 39.45, f8 39.13 dB.

## What the activation half costs: **1.48 dB and 0.0036 SSIM**

Weights INT8 alone: **40.22 dB / 0.9708** (byte-identical to the earlier committed run,
`gate1_psnr_20261116_051022` — the same seed and path reproduce to 2 decimals, so this leg is
deterministic). Adding the INT8 activation operand at the measured per-layer SA:
**38.73 dB / 0.9672** — still 10.7 dB above the 28 dB bar, so **both operands quantised PASS**.

The cost is not uniformly spread across the 9 frames — it lands on the frames that had the most
headroom:

| frame | weight-only | + INT8 activations | Δ |
|---|---|---|---|
| f0 | 53.49 dB | 44.02 dB | **−9.47** |
| f1 | 38.47 | 35.65 | −2.82 |
| f2 | 43.21 | 41.31 | −1.90 |
| f3 | 41.85 | 39.79 | −2.07 |
| f4 | 40.89 | 39.73 | −1.17 |
| f5 | 37.88 | 37.29 | −0.59 |
| f6 | 39.33 | 38.99 | −0.34 |
| f7 | 39.45 | 38.22 | −1.23 |
| f8 | 39.13 | 38.40 | −0.73 |

f0 is the causal boundary frame — trivially predictable (53.49 dB weight-only) and therefore the
frame with the smallest absolute error to begin with, so 8-bit activation noise dominates it. The
steady-state frames (f4–f8) lose 0.3–1.2 dB. Consequence for the paper: quote the decode quality
against the **steady-state** frames if a single number is wanted, and keep the weight-only /
weight+activation pair side by side — they are different claims.

## How the INT8 activation was applied (so the claim is auditable)

`phase1_quant/psnr_gate1.py --act-scales <calibrate.py json>` installs a pre-hook on each of the 36
decoder layers that replaces the layer input `x` with `clamp(round(x/SA), -127, 127)·SA`, with
`SA` that layer's measured value, and asserts that **36/36 hooks fired** — a scale table that
silently misses a layer is a hard error, not a row of plausible numbers. One SA per layer (the max
over the decode's frame chunks), which is what a single NNAPI graph can carry.

Not modelled, and not claimed: per-frame/per-call SA (each layer's chunks vary by up to 1.97x, so a
per-chunk table would do slightly better), the INT32 bias in `SA*SW` units, and the residual/SiLU
wiring of the device walk. Those affect the **device** decode; this number prices the arithmetic.
