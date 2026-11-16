# Calibration — `SA = max|x| / 127`, the activation scale (deliverable 4)

built `20261116_063707` on `.venv-ref` (torch 2.15.0+cpu), host only, no device involved.

## VAE decoder — 36 layers

latent `[1, 16, 3, 34, 60]` seed 0; every layer's scale is measured on the activation its own taps consumed.

`SA` range **0.004822 … 0.252453** (spread 52.36x) against the placeholder `1/127 = 0.007874`.

| layer | in shape | max&#124;x&#124; | SA_in | SA_out | out/in | max&#124;x_out&#124; |
|---|---|---|---|---|---|---|
| `conv_in` | [1, 16, 1, 34, 60] | 2.522 | 0.019856 | 0.012566 | 0.63x | 1.596 |
| `mid_block.resnets.0.conv1` | [1, 384, 1, 34, 60] | 4.653 | 0.036640 | 0.431392 | 11.77x | 54.787 |
| `mid_block.resnets.0.conv2` | [1, 384, 1, 34, 60] | 0.612 | 0.004822 | 0.074764 | 15.51x | 9.495 |
| `mid_block.resnets.1.conv1` | [1, 384, 1, 34, 60] | 2.635 | 0.020750 | 0.094050 | 4.53x | 11.944 |
| `mid_block.resnets.1.conv2` | [1, 384, 1, 34, 60] | 0.776 | 0.006107 | 0.028660 | 4.69x | 3.640 |
| `up_blocks.0.resnets.0.conv1` | [1, 384, 1, 34, 60] | 6.109 | 0.048100 | 0.062188 | 1.29x | 7.898 |
| `up_blocks.0.resnets.0.conv2` | [1, 384, 1, 34, 60] | 1.229 | 0.009673 | 0.026496 | 2.74x | 3.365 |
| `up_blocks.0.resnets.1.conv1` | [1, 384, 1, 34, 60] | 5.045 | 0.039722 | 0.049370 | 1.24x | 6.270 |
| `up_blocks.0.resnets.1.conv2` | [1, 384, 1, 34, 60] | 5.428 | 0.042738 | 0.036527 | 0.85x | 4.639 |
| `up_blocks.0.resnets.2.conv1` | [1, 384, 1, 34, 60] | 4.341 | 0.034180 | 0.056225 | 1.64x | 7.141 |
| `up_blocks.0.resnets.2.conv2` | [1, 384, 1, 34, 60] | 4.782 | 0.037653 | 0.030339 | 0.81x | 3.853 |
| `up_blocks.0.upsamplers.0.time_conv` | [1, 384, 1, 34, 60] | 7.630 | 0.060082 | 0.057658 | 0.96x | 7.323 |
| `up_blocks.0.upsamplers.0.resample.1` | [2, 384, 68, 120] | 7.469 | 0.058815 | 0.252453 | 4.29x | 32.062 |
| `up_blocks.1.resnets.0.conv1` | [1, 192, 2, 68, 120] | 15.479 | 0.121884 | 0.114914 | 0.94x | 14.594 |
| `up_blocks.1.resnets.0.conv2` | [1, 384, 2, 68, 120] | 4.158 | 0.032742 | 0.042721 | 1.30x | 5.426 |
| `up_blocks.1.resnets.0.conv_shortcut` | [1, 192, 2, 68, 120] | 32.062 | 0.252453 | 0.069312 | 0.27x | 8.803 |
| `up_blocks.1.resnets.1.conv1` | [1, 384, 2, 68, 120] | 5.334 | 0.042004 | 0.064573 | 1.54x | 8.201 |
| `up_blocks.1.resnets.1.conv2` | [1, 384, 2, 68, 120] | 6.111 | 0.048119 | 0.060881 | 1.27x | 7.732 |
| `up_blocks.1.resnets.2.conv1` | [1, 384, 2, 68, 120] | 4.581 | 0.036074 | 0.050575 | 1.40x | 6.423 |
| `up_blocks.1.resnets.2.conv2` | [1, 384, 2, 68, 120] | 6.132 | 0.048280 | 0.062944 | 1.30x | 7.994 |
| `up_blocks.1.upsamplers.0.time_conv` | [1, 384, 2, 68, 120] | 14.045 | 0.110594 | 0.090739 | 0.82x | 11.524 |
| `up_blocks.1.upsamplers.0.resample.1` | [4, 384, 136, 240] | 11.524 | 0.090739 | 0.150476 | 1.66x | 19.110 |
| `up_blocks.2.resnets.0.conv1` | [1, 192, 4, 136, 240] | 4.171 | 0.032841 | 0.047712 | 1.45x | 6.059 |
| `up_blocks.2.resnets.0.conv2` | [1, 192, 4, 136, 240] | 6.661 | 0.052448 | 0.076883 | 1.47x | 9.764 |
| `up_blocks.2.resnets.1.conv1` | [1, 192, 4, 136, 240] | 3.872 | 0.030490 | 0.032778 | 1.08x | 4.163 |
| `up_blocks.2.resnets.1.conv2` | [1, 192, 4, 136, 240] | 6.743 | 0.053093 | 0.079617 | 1.50x | 10.111 |
| `up_blocks.2.resnets.2.conv1` | [1, 192, 4, 136, 240] | 3.155 | 0.024841 | 0.039665 | 1.60x | 5.037 |
| `up_blocks.2.resnets.2.conv2` | [1, 192, 4, 136, 240] | 5.722 | 0.045058 | 0.099491 | 2.21x | 12.635 |
| `up_blocks.2.upsamplers.0.resample.1` | [4, 192, 272, 480] | 27.088 | 0.213290 | 0.239802 | 1.12x | 30.455 |
| `up_blocks.3.resnets.0.conv1` | [1, 96, 4, 272, 480] | 3.538 | 0.027856 | 0.026978 | 0.97x | 3.426 |
| `up_blocks.3.resnets.0.conv2` | [1, 96, 4, 272, 480] | 4.834 | 0.038064 | 0.063959 | 1.68x | 8.123 |
| `up_blocks.3.resnets.1.conv1` | [1, 96, 4, 272, 480] | 3.756 | 0.029578 | 0.032248 | 1.09x | 4.095 |
| `up_blocks.3.resnets.1.conv2` | [1, 96, 4, 272, 480] | 5.942 | 0.046785 | 0.060177 | 1.29x | 7.643 |
| `up_blocks.3.resnets.2.conv1` | [1, 96, 4, 272, 480] | 4.214 | 0.033179 | 0.032101 | 0.97x | 4.077 |
| `up_blocks.3.resnets.2.conv2` | [1, 96, 4, 272, 480] | 5.321 | 0.041895 | 0.080382 | 1.92x | 10.208 |
| `conv_out` | [1, 96, 4, 272, 480] | 3.884 | 0.030584 | 0.011217 | 0.37x | 1.425 |

### Post-add range per resnet (the tensor the residual add produces)

`conv2`'s own range is **not** the resnet's output range: the add of `conv2_out + shortcut_out` raises it by 1.03x … 4.76x here, so a residual add quantised at `conv2`'s scale would clip.

| resnet | post-add max&#124;x&#124; | SA | ÷ conv2's own range |
|---|---|---|---|
| `mid_block.resnets.0` | 9.785 | 0.077050 | 1.03x |
| `mid_block.resnets.1` | 8.452 | 0.066555 | 2.32x |
| `up_blocks.0.resnets.0` | 8.439 | 0.066452 | 2.51x |
| `up_blocks.0.resnets.1` | 8.425 | 0.066341 | 1.82x |
| `up_blocks.0.resnets.2` | 7.630 | 0.060082 | 1.98x |
| `up_blocks.1.resnets.0` | 10.676 | 0.084059 | 1.97x |
| `up_blocks.1.resnets.1` | 10.755 | 0.084683 | 1.39x |
| `up_blocks.1.resnets.2` | 14.045 | 0.110594 | 1.76x |
| `up_blocks.2.resnets.0` | 20.045 | 0.157834 | 2.05x |
| `up_blocks.2.resnets.1` | 18.880 | 0.148658 | 1.87x |
| `up_blocks.2.resnets.2` | 27.088 | 0.213290 | 2.14x |
| `up_blocks.3.resnets.0` | 33.880 | 0.266771 | 4.17x |
| `up_blocks.3.resnets.1` | 36.348 | 0.286205 | 4.76x |
| `up_blocks.3.resnets.2` | 42.673 | 0.336008 | 4.18x |

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

