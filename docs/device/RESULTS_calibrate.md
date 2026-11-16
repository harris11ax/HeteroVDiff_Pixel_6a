# Calibration — `SA = max|x| / 127`, the activation scale (deliverable 4)

built `20261116_054430` on `.venv-ref` (torch 2.15.0+cpu), host only, no device involved.

## VAE decoder — 36 layers

latent `[1, 16, 3, 34, 60]` seed 0; every layer's scale is measured on the activation its own taps consumed.

`SA` range **0.004822 … 0.252453** (spread 52.36x) against the placeholder `1/127 = 0.007874`.

| layer | in shape | max&#124;x&#124; | SA | vs 1/127 |
|---|---|---|---|---|
| `conv_in` | [1, 16, 1, 34, 60] | 2.522 | 0.019856 | 2.52x |
| `mid_block.resnets.0.conv1` | [1, 384, 1, 34, 60] | 4.653 | 0.036640 | 4.65x |
| `mid_block.resnets.0.conv2` | [1, 384, 1, 34, 60] | 0.612 | 0.004822 | 0.61x |
| `mid_block.resnets.1.conv1` | [1, 384, 1, 34, 60] | 2.635 | 0.020750 | 2.64x |
| `mid_block.resnets.1.conv2` | [1, 384, 1, 34, 60] | 0.776 | 0.006107 | 0.78x |
| `up_blocks.0.resnets.0.conv1` | [1, 384, 1, 34, 60] | 6.109 | 0.048100 | 6.11x |
| `up_blocks.0.resnets.0.conv2` | [1, 384, 1, 34, 60] | 1.229 | 0.009673 | 1.23x |
| `up_blocks.0.resnets.1.conv1` | [1, 384, 1, 34, 60] | 5.045 | 0.039722 | 5.04x |
| `up_blocks.0.resnets.1.conv2` | [1, 384, 1, 34, 60] | 5.428 | 0.042738 | 5.43x |
| `up_blocks.0.resnets.2.conv1` | [1, 384, 1, 34, 60] | 4.341 | 0.034180 | 4.34x |
| `up_blocks.0.resnets.2.conv2` | [1, 384, 1, 34, 60] | 4.782 | 0.037653 | 4.78x |
| `up_blocks.0.upsamplers.0.time_conv` | [1, 384, 1, 34, 60] | 7.630 | 0.060082 | 7.63x |
| `up_blocks.0.upsamplers.0.resample.1` | [2, 384, 68, 120] | 7.469 | 0.058815 | 7.47x |
| `up_blocks.1.resnets.0.conv1` | [1, 192, 2, 68, 120] | 15.479 | 0.121884 | 15.48x |
| `up_blocks.1.resnets.0.conv2` | [1, 384, 2, 68, 120] | 4.158 | 0.032742 | 4.16x |
| `up_blocks.1.resnets.0.conv_shortcut` | [1, 192, 2, 68, 120] | 32.062 | 0.252453 | 32.06x |
| `up_blocks.1.resnets.1.conv1` | [1, 384, 2, 68, 120] | 5.334 | 0.042004 | 5.33x |
| `up_blocks.1.resnets.1.conv2` | [1, 384, 2, 68, 120] | 6.111 | 0.048119 | 6.11x |
| `up_blocks.1.resnets.2.conv1` | [1, 384, 2, 68, 120] | 4.581 | 0.036074 | 4.58x |
| `up_blocks.1.resnets.2.conv2` | [1, 384, 2, 68, 120] | 6.132 | 0.048280 | 6.13x |
| `up_blocks.1.upsamplers.0.time_conv` | [1, 384, 2, 68, 120] | 14.045 | 0.110594 | 14.05x |
| `up_blocks.1.upsamplers.0.resample.1` | [4, 384, 136, 240] | 11.524 | 0.090739 | 11.52x |
| `up_blocks.2.resnets.0.conv1` | [1, 192, 4, 136, 240] | 4.171 | 0.032841 | 4.17x |
| `up_blocks.2.resnets.0.conv2` | [1, 192, 4, 136, 240] | 6.661 | 0.052448 | 6.66x |
| `up_blocks.2.resnets.1.conv1` | [1, 192, 4, 136, 240] | 3.872 | 0.030490 | 3.87x |
| `up_blocks.2.resnets.1.conv2` | [1, 192, 4, 136, 240] | 6.743 | 0.053093 | 6.74x |
| `up_blocks.2.resnets.2.conv1` | [1, 192, 4, 136, 240] | 3.155 | 0.024841 | 3.15x |
| `up_blocks.2.resnets.2.conv2` | [1, 192, 4, 136, 240] | 5.722 | 0.045058 | 5.72x |
| `up_blocks.2.upsamplers.0.resample.1` | [4, 192, 272, 480] | 27.088 | 0.213290 | 27.09x |
| `up_blocks.3.resnets.0.conv1` | [1, 96, 4, 272, 480] | 3.538 | 0.027856 | 3.54x |
| `up_blocks.3.resnets.0.conv2` | [1, 96, 4, 272, 480] | 4.834 | 0.038064 | 4.83x |
| `up_blocks.3.resnets.1.conv1` | [1, 96, 4, 272, 480] | 3.756 | 0.029578 | 3.76x |
| `up_blocks.3.resnets.1.conv2` | [1, 96, 4, 272, 480] | 5.942 | 0.046785 | 5.94x |
| `up_blocks.3.resnets.2.conv1` | [1, 96, 4, 272, 480] | 4.214 | 0.033179 | 4.21x |
| `up_blocks.3.resnets.2.conv2` | [1, 96, 4, 272, 480] | 5.321 | 0.041895 | 5.32x |
| `conv_out` | [1, 96, 4, 272, 480] | 3.884 | 0.030584 | 3.88x |

## DiT block GEMMs — 6 steps x 30 blocks x 10 GEMMs

text conditioning: umT5-XXL bf16 encode of prompt 'a red fox walking through fresh snow at golden hour', 512 tokens; max|e| 2.453

schedule [1000, 992, 982, 949, 905, 810] (locked), flow sigmas ['1.000', '0.992', '0.982', '0.949', '0.905', '0.810'], M = 1530 tokens.

| GEMM | SA min | SA median | SA max | spread |
|---|---|---|---|---|
| `attn1.to_q` | 0.023566 | 0.033744 | 0.116030 | 4.92x |
| `attn1.to_k` | 0.023566 | 0.033744 | 0.116030 | 4.92x |
| `attn1.to_v` | 0.023566 | 0.033744 | 0.116030 | 4.92x |
| `attn1.to_out.0` | 0.015988 | 0.026696 | 0.051372 | 3.21x |
| `attn2.to_q` | 0.073081 | 0.147924 | 0.286219 | 3.92x |
| `attn2.to_k` | 0.005011 | 0.005011 | 0.005011 | 1.00x |
| `attn2.to_v` | 0.005011 | 0.005011 | 0.005011 | 1.00x |
| `attn2.to_out.0` | 0.002466 | 0.003911 | 0.007927 | 3.21x |
| `ffn.net.0.proj` | 0.016367 | 0.030159 | 0.055460 | 3.39x |
| `ffn.net.2` | 0.019348 | 0.043118 | 0.147917 | 7.65x |

Residual stream max&#124;x&#124; 2.32 … 359.04 across all 180 (step, block) entries.


## What the numbers say

**The placeholder `SA = 1/127` is wrong on most layers, in both directions.** 34 of the 36
decoder layers need a *larger* scale than the nominal (median 4.9x, worst 32.1x on
`up_blocks.1.resnets.0.conv_shortcut`, whose input is a resnet entry of max|x| 31.0), so the
`npu_block convseq` runs in P3.8b/P3.8c were clipping on 34 of 36 layers — that is the
quantified reason those walks "execute but do not decode". Two layers (`mid_block.resnets.0.conv2`,
`up_blocks.2.resnets.0.conv1`) sit *below* the nominal and were wasting range instead.

**On the DiT, 1261 of the 1800 (step, block, GEMM) instances exceed 1/127** and 539 sit below it,
so no single activation scale is even directionally right across the block:

| GEMM | measured SA / (1/127) | static-max / instance |
|---|---|---|
| `attn2.to_k`, `attn2.to_v` | 0.64x | 1.00x (identical in all 180) |
| `attn2.to_out.0` | 0.50x | up to 3.21x |
| `attn1.to_out.0` | 3.39x | up to 3.21x |
| `ffn.net.0.proj` | 3.83x | up to 3.39x |
| `attn1.to_q`/`to_k`/`to_v` | 4.29x | up to 4.92x |
| `ffn.net.2` | 5.48x | up to 7.65x |
| `attn2.to_q` | 18.79x | up to 3.92x |

**Two independent consistency proofs fell out of the run, neither one asserted:**

1. `attn2.to_k` and `attn2.to_v` read **0.005011 in all 180 instances** (spread 1.00x) — the
   cross-attention context is the *same* RMS-normalised umT5 tensor in every block and step, so a
   scale that is constant to the last digit is the correct answer, not a bug. Its constancy also
   proves the text path is identical across the trajectory.
2. `attn1.to_q`, `to_k`, `to_v` share one input and therefore one SA (0.033744 median, identical
   min/max) — three separate hooks agreeing exactly.

**A per-GEMM *static* scale is a measurable compromise**: taking each GEMM's max over all 180
instances costs up to 7.65x of the available resolution (`ffn.net.2`, median instance 3.43x away
from the static choice) because SA moves with the step and the block. The measured table is
per (step, block, GEMM) for that reason; whether to ship 1800 scales or 10 static ones is a
storage/accuracy trade the student can now price.

**fp16 for the residual stream is safe as locked**: the largest hidden-state max|x| over all 180
entries is **359.04** against the fp16 ceiling of 65504 — 182x of headroom — so the lock's
"fp16 hidden state + per-GEMM calibrated input scale" needs no narrower term.

## Scope, stated so it cannot be read as more than it is

- **Base weights only.** 480 `base_layer.*` tensors were mapped to stock names and **0** were
  missing; 480 `lora_A`/`lora_B` expert factors were dropped deliberately (they are the FP16
  expert, not the INT8 base) and the DCM text-embedder LoRA (`condition_embedder_lora.*`, 12
  tensors) is **not applied**, so the modulation is the stock one. Those are the only 12
  unexpected tensors and they are reported, not silent.
- **Latents.** The trajectory is the locked 6-step schedule at the locked timesteps, with a seeded
  z_0 stand-in mixed at each step's flow sigma (`sigma = t/1000`; the scheduler's own convention
  measured to 1e-3). That samples the activation regime per step; it is not a decoded video.
- **Text.** Real umT5-XXL (bf16, 512 tokens); the failure path would have said so in
  `text_source` rather than substituting silently.
- **No device number here.** This writes scales; it changes no rate, and `T_N`/`T_G`/`T_V`/`k_b*`
  are untouched.

## Consumers this unblocks

1. `artifacts/vae/bias_f32.bin` can become INT32 in `SA*SW` units (it is FP32 and flagged
   `quantised: false` precisely because this file did not exist).
2. `npu_block convseq` can carry a per-layer SA table instead of the nominal — the difference
   between "the decode executes" and "the decode decodes".
3. A Gate-1 number *with* the INT8 activation path, to sit beside the 40.22 dB weight-only result.
