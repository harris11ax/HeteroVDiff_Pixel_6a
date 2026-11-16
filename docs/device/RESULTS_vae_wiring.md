# P3.11 — the decode's wiring: the norm between two convs, measured

*2026-11-16 06:36. Host-only. Tools `phase1_quant/calibrate.py` (extended),
`phase1_quant/vae_sa.py` (extended), new `phase1_quant/vae_wiring.py`,
`phase1_quant/check_norm_path.py`. Artifacts `artifacts/vae/norms_f32.bin`,
`artifacts/vae/vae_wiring.json`, `artifacts/vae/act_scales2_f32.bin`.
No device number is produced, changed or claimed here; `T_V`, `T_N`, `T_G` and
`k_b*` are untouched.*

## Why

P3.10 left this as the next action: *"`calibrate.py` must record the per-layer
OUTPUT range (its `max_abs_out` reads a flat 1.0, so it was never a real output
calibration) → table becomes 2 × 36 floats, and wire the residual/SiLU/clamp
path — that is what turns 'executes' into 'decodes'."*

That is now measured, and measuring it answered a question the project did not
know it had: **the activation chain through the VAE decoder is not a rescale
between consecutive convs**, so the wiring `convseq` implements (and the
"byte-copy wiring" assumption in every earlier note) is wrong by up to 89×.

## 1. The output range exists, and it is not the input range

`calibrate.py` now registers a **forward** hook per layer alongside the pre-hook
and keeps the max across the decode's frame chunks. `resnets[].max_abs_out` is
the **post-add** tensor (`conv2_out + shortcut_out`), which no conv owns.

| quantity | range | source |
|---|---|---|
| `SA_in` (per-layer input scale) | 0.004822 … 0.252453 | pre-hook, 36/36 layers |
| `SA_out` (per-layer output scale) | 0.011217 … 0.431392 | forward hook |
| `SA_out / SA_in` | **0.275 … 15.505** (median **1.305**) | derived |
| post-add range ÷ conv2's own range | **1.03× … 4.76×** (median 1.97×) | 14 resnets |

Consequence for the artifact: `act_scales2_f32.bin` = **288 B = 2 × 36 f32**
(`SA_in[36] ++ SA_out[36]`), verified float-identical to the calibration
(delta 0). `act_scales_f32.bin` (144 B, `SA_in` only) is kept because the
currently-built `convseq` binary reads it; one source, two views, one tool.

## 2. The 89× step is a norm, and it is now proved

`mid_block.resnets.0.conv1` emits max|x| **54.79**; the next conv in artifact
order, `...resnets.0.conv2`, consumes max|x| **0.612** — 89× smaller. No
requant does that. `WanResidualBlock.forward` puts `norm2` (`WanRMS_norm`) and
SiLU between them, and `WanRMS_norm` is
`F.normalize(x, dim=1) * sqrt(C) * gamma + bias` — an **L2 normalisation across
the channel axis at every (b,t,h,w)**, so it rescales per pixel, not per channel
or per tensor. `bias` is the python float `0.0` (`images=False`), so `gamma`
[C,1,1,1] is the only parameter.

`check_norm_path.py` captures conv1's real output in a decode, applies that norm
and SiLU by hand, and compares against the number the calibration measured on
conv2's *input*:

| step | max&#124;x&#124; |
|---|---|
| `conv1` output (one call) | 47.198 |
| after `norm2` | 4.5325 |
| after `SiLU` | **0.6124** |
| calibration's measured `conv2` INPUT | **0.6124** |
| relative difference | **0.00 %** |

**PASS.** The norm is the missing op, and the ranges in `vae_wiring.json` are
the right thing to implement. (conv1's output reads 47.198 here versus the
calibration's 54.787 because this hook keeps the first call and the calibration
keeps the max over all three chunks — same tensor, different aggregation.)

## 3. The plan (`artifacts/vae/vae_wiring.json`, 128 ops)

Walked structurally off the instantiated model, not retyped:

| op | count | note |
|---|---|---|
| `conv` | 36 | **36/36** artifact layers found — the walk and the artifact agree as sets |
| `norm` | 29 | 14 resnets × `norm1`+`norm2`, plus `norm_out`; 7,776 gamma floats = **31,104 B** |
| `silu` | 29 | one after each norm |
| `save` + `add` | 14 + 14 | the residual add; each add carries its `post_add_sa` |
| `temporal_shuffle` | 2 | `time_conv` 2C→C over t: `reshape(b,2,c,t,h,w) → stack dim3 → (b,c,2t,h,w)` |
| `upsample2d` | 3 | nearest-exact ×2 before each `resample` conv |
| `attn` | 1 | mid-block; not a conv, no INT8 artifact, priced on Mali (0.0855 s) |

`norms_f32.bin`: **29 gammas, 7,776 floats, 31,104 B**, every byte re-read and
compared float-for-float with the module parameter (delta 0), and none of them
all-ones (a checkpoint that failed to load would give exactly that).

Two cross-checks fell out rather than being asserted:

- the post-add range of `up_blocks.{0,1}.resnets.2` is **exactly** the input
  range the calibration measured on the following `time_conv` (0.06008 and
  0.11059 in both) — i.e. the add's output *is* the consumer's input, which is
  what makes those two hooks agree to the last digit;
- `time_conv` → `resample.1` has no norm between them, and their measured ranges
  agree to **0.98 / 1.00**, versus 0.17–89 across a norm. The ratio tracks the
  presence of a norm, not the layer.

### Ordering defect found (recorded, not silently fixed)

The artifact indexes each resnet as `conv1, conv2, conv_shortcut`, but
`forward` applies the shortcut **first** (`h = self.conv_shortcut(x)`). So the
artifact's layer index is a **storage order, not an execution order**; a device
walking `SA_in[i]`, `SA_in[i+1]` by index is walking the table out of order at
every resnet that owns a shortcut. Exactly **3 positions differ**, all in
`decoder.up_blocks.1.resnets.0` (artifact 13/14/15 = conv1/conv2/shortcut;
execution order = shortcut/conv1/conv2). `vae_wiring.json` now carries
`artifact_index` on every conv op so the two orders cannot be confused again.

## 4. What this does NOT do

Nothing is wired into `convseq` yet: the norm, SiLU, temporal shuffle, 2×
upsample and the 14 residual adds are all still unexecuted on the device, so
**the decode still executes without decoding** and no quality claim exists. The
harness epilogue (P3.10: 12,028 ms = 52.9 % of the 22,730 ms walk) is unaffected
by this tick.

Why the epilogue cannot simply be folded: the norm's scale is **per pixel**, so
neither the norm nor SiLU can be pushed into a conv's `FUSED_NONE` epilogue. The
integer-only epilogue only becomes possible *after* a norm, where producer and
consumer can share one scale.

## Next action

Implement the plan in `convseq` (one bounded device unit): per-layer
`sa_in`/`sa_out` from `act_scales2_f32.bin`, the L2-over-channels norm with
`norms_f32.bin` gammas, SiLU, the shuffle and the residual adds at
`post_add_sa`. Then a floor-started run for the walk time, and only then is
"the decode decodes" a thing that can be checked against Gate 1.
