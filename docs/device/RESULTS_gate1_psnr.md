# Gate 1 — end-to-end PSNR/SSIM: FP32 vs the INT8 artifact's weights

`20261116_051022` · latent `[1, 16, 3, 34, 60]` seed 0 · **weight-only** (quantise-dequantise; activations stay FP32 because `SA` is `calibrate.py`'s output).

| config | layers quantised | PSNR | SSIM | verdict |
|---|---|---|---|---|
| all quantised | 36/36 | **40.22 dB** | **0.9708** | **PASS** |
| quantised except worst 2 by weight cosine | 34/36 | **48.99 dB** | **0.9962** | **PASS** |

Gates: PSNR ≥ 28.0 dB, SSIM ≥ 0.92.

Round-trip: every rebuilt weight reproduces the artifact manifest's recorded per-layer relFro (max |diff| 4.5e-07), so the OHWI tap inversion is the artifact and not a plausible-looking transpose.

Per-frame PSNR (all-quantised): f0 53.49, f1 38.47, f2 43.21, f3 41.85, f4 40.89, f5 37.88, f6 39.33, f7 39.45, f8 39.13 dB.
