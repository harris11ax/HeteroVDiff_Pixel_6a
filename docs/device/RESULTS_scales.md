# Per-tensor vs per-output-column INT8 scales — the parity cost, measured

*2026-11-15. Host-side parity on the real FP32 checkpoint. Script:
`phase1_quant/scale_parity.py`. Artifacts: `docs/device/scale_parity_block0.json`,
`docs/device/scale_parity_blocks.json` (blocks 5, 15, 29).*

## Why this exists

The locked quantisation is **per-output-column symmetric INT8**, but NNAPI's
`FULLY_CONNECTED` rejects `TENSOR_QUANT8_SYMM_PER_CHANNEL` at `addOperation`
(`RESULTS_npu_tn.md`, finding #2 — measured, not assumed). The NPU stage can
therefore only express **one scale per weight matrix**. That left a fork with the
student: accept the parity cost on the NPU stage, or restrict the NPU to shapes
with an equivalent path. This document measures the cost so the fork is a
comparison, not a guess. **Nothing here decides it.**

## Method

- Weights: the actual FP32 tensors of `diffusion_pytorch_model.safetensors`
  (`[N_out, K]`, row-major — the stored layout), read by mmap, **no INT8 artifact
  written** (the single-stored-copy rule is untouched).
- Layouts compared: per-tensor `s = max|W|/127`, and per-column `s_i = max|W[i,:]|/127`.
  Both symmetric, both round-to-nearest, clamped to ±127.
- Reference arithmetic in **float64**, so the reported error is the quantiser's.
- Activation: seeded standard normal, 256 tokens, quantised INT8 with **one
  per-tensor activation scale** — exactly what the device FC path does, in *both*
  layouts, so the delta is the weight-scale layout and nothing else.
- Sample: **40 GEMM instances** = blocks 0, 5, 15, 29 × the block's 10 GEMMs.
  Every block is shape-identical (49,586,432 params each), so this measures
  whether the penalty is a shape property or a weight property.

## Result — the penalty is real, and it is not uniform

| metric (40 GEMM instances) | per-tensor | per-column |
|---|---|---|
| weight rel. Frobenius error, min / med / max | 1.00 / 2.24 / **7.90 %** | 0.67 / 0.85 / 1.22 % |
| weight cosine, min / med | **0.996892** / 0.999750 | 0.999926 / 0.999964 |
| GEMM output rel. error, min / med / max | 1.48 / 2.52 / **7.99 %** | 1.28 / 1.37 / 1.65 % |
| GEMM output cosine, min / med | **0.996820** / 0.999682 | 0.999864 / 0.999907 |
| instances failing the Gate-1 cosine ≥ 0.999 bar | **5 / 40 (W)**, **6 / 40 (Y)** | 0 / 40 |

MAC-weighted GEMM output error: **per-tensor 3.75 %** (block 0) / **3.25 %** (blocks
5,15,29) vs **per-column 1.46 %** in both samples. Mean over the 40 instances
**3.01 % vs 1.40 % → 2.15× mean penalty, median 1.74×, worst 5.44×**.

Per-shape, averaged over the four blocks (output rel. error, %):

| GEMM | per-tensor (max) | per-column | penalty |
|---|---|---|---|
| `ffn.up` `[8960,1536]` | 4.39 (6.86) | 1.42 | **3.10×** |
| `ffn.down` `[1536,8960]` | 3.12 (3.50) | 1.62 | 1.92× |
| `attn1.to_q` | 3.72 (5.06) | 1.38 | 2.70× |
| `attn1.to_k` | 4.54 (**7.99**) | 1.45 | **3.14×** |
| `attn1.to_v` | 2.69 (4.45) | 1.46 | 1.84× |
| `attn1.to_out` | 2.15 (2.54) | 1.39 | 1.55× |
| `attn2.to_q` | 1.92 (2.61) | 1.37 | 1.40× |
| `attn2.to_k` | 1.61 (1.78) | 1.30 | **1.24×** |
| `attn2.to_v` | 1.58 (1.70) | 1.30 | **1.21×** |
| `attn2.to_out` | 4.42 (7.25) | 1.34 | **3.30×** |

## Mechanism (not a guess — it correlates)

The per-column scale spread `max(s_i)/min(s_i)` within a matrix predicts the
per-tensor penalty with **r = 0.805**; per-column error is flat against the same
spread (**r = 0.101**). One global scale is fixed by the widest column, and every
narrow column is quantised with a step far too coarse for its range. The worst
instances are exactly the high-spread ones (`blocks.29.attn2.to_out`, spread 20.8×,
7.25 %); the cheapest are low-spread (`attn2.to_k`, 2.0–3.0×, 1.2× penalty).

## Consequence for the fork (the comparison, not the verdict)

- **Branch A — NPU stage runs per-tensor scales:** costs **2.15× the mean GEMM
  output error** (worst instance 5.44×) and breaks the Gate-1 `cosine ≥ 0.999`
  weight bar on **5 of 40** instances (worst 0.996892). It never breaks the
  per-column error floor (1.65 % max).
- **Branch B — NPU restricted to shapes with an equivalent path:** the penalty is a
  **weight property, not a shape property** — `attn1.to_k` is 3.14× on average and
  holds the worst single instance in the whole 40-GEMM sample (blocks.29, 7.99 %),
  while `attn2.to_k` is 1.24× in the same four blocks. So a shape-only restriction
  cannot be justified from this data; the only defensible restriction is
  per-tensor-where-cheap, which covers `attn2.to_k` + `attn2.to_v` = **10.2 % of
  the block's MACs** (4.72 M of 46.40 M per token) and leaves 89.8 % of the
  block's MACs unable to use the NPU under the locked quantisation.

## What this does NOT measure

- **No device latency** — nothing here is a rate; T_N/T_G/T_V are untouched.
- **No end-to-end Gate 1** (PSNR/SSIM on decoded frames): that needs the
  quantiser + FP32 reference path, still outstanding.
- Activations are synthetic. Real calibrated activation scales would move the
  absolute error, but both layouts share the same activation treatment, so the
  *delta* between layouts is layout-only.
- Whether the NPU's own accumulator is exact int32 (Mali's was verified exact;
  the NPU's per-tensor path was only verified *compilable*). This delta would be
  identical under exact int32 accumulation in either engine.
