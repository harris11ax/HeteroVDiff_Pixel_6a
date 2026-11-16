# P3.18 — the glue is not the deficit: every rounding hop is worth ≤ 0.013 dB

*2026-11-16 10:28–10:51. Host-only: **no device run, no device rate touched** — `T_V` 8.908 s,
`T_N` exact 0.8905 s/block, `k_b*` 165 are exactly as committed. Instruments: `tools/npu/vae_sim.py
--glue` / `--convout` (new arms + per-hop instrumentation) and a new tool
`tools/npu/vae_gather_probe.py` (P3.19). Captures `vae_glue_20261116_102817`,
`vae_convout_20261116_104708`, `vae_gather2_20261116_105111`; latent `dump_20261116_080721`
(the post-fix device walk's own latent), same artifact and scales as P3.14–P3.17.*

## What was unrun, and why it had to be run

P3.17 ended with the arithmetic `38.73 dB (Gate 1, per-layer) → 34.34 (oracle: every conv fed the
reference's own input) → 20.59 (wire)`, i.e. **4.39 dB conv-intrinsic + 13.75 dB "glue", no
residual**. The word "glue" there was a *bucket*, not a measurement: it held the norm+SiLU hops,
the residual adds, the conv's input build, and the conv's own output requant, all lumped together
with the propagation of every earlier error. This run prices each of them separately.

## Result 1 — removing EVERY glue int8 hop is worth +0.013 dB

Three new arms, same convs, same weights, same per-tap scheme; only which glue ops round to int8
changes (`gluefp` runs `NORMSILU`/`NORM`/`SILU` and the residual `ADD` in float32 — the state then
carries scale 1.0, so the only remaining requant is the conv's input build).

| arm | glue in float32 | final PSNR (2.0) | cosine | rel-RMSE | worst conv | int8 clamps |
|---|---|---|---|---|---|---|
| **`wire`** | — | **20.5876 dB** | **0.888501** | 47.12 % | 34 (0.714915) | 553,689 |
| **`gluefp`** | norm+add | 20.6010 dB | 0.888720 | 47.05 % | 34 (0.714984) | 434,518 |
| **`normfp`** | norm | 20.5886 dB | 0.888522 | 47.12 % | 34 (0.714933) | 528,383 |
| **`addfp`** | add | 20.6015 dB | 0.888737 | 47.05 % | 34 (0.714992) | 460,294 |

`wire` reproduces P3.17 **to the digit** (20.5876 / 0.888501 / 34 at 0.714915), so the arms are
comparable, and the instrumentation did not perturb the arithmetic. Deleting every glue
quantisation in a 36-conv decode buys **+0.013 dB and 0.0002 of cosine**. The 13.75 dB does not
live there.

## Result 2 — the one-hop census: 1.96 pt of loss across 79 sites, and it does not compound

`hop` = 1 − cos(fp32 result of the op, that same result requantised at the op's own `out_sa`),
priced **at the input the walk handed it**.

| class | ops | Σ hop | mean | max | worst site |
|---|---|---|---|---|---|
| `INPUT-BUILD` (conv input: shuffle/upsample + requant to `SA_in`) | 36 | **0.0035 pt** | 0.0001 pt | 0.0035 pt | after conv 12 |
| `NORMSILU` | 29 | 1.746 pt | 0.060 pt | **0.529 pt** | after conv 3 |
| `ADD` (residual) | 14 | 0.211 pt | 0.015 pt | 0.085 pt | after conv 10 |

Two facts worth keeping. **The conv's input requant is free** — the artifact's `SA_in[c]` equals
the producer's `SA_out` to ratio **1.000** at 35 of 36 sites (0.980 at conv 12), so there is
nothing to rescale and the hop costs 0.0001 pt. And the **largest single hop in the decode is a
calibration defect, not a rounding one**: the `NORMSILU` after conv 3 (`mid_block.resnets.1.conv1`)
quantises post-SiLU values on a grid of `out_sa` 0.0061065 while its own input was at 0.09405 —
**0.97 % of its codes clamp** at 127 (the next-worst norm hop is 0.058 %, the rest ≤ 0.002 %),
which is 0.53 pt of this table's 1.96 pt. That is `calibrate.py`'s per-tensor max being measured
on the wrong pass at one site; it is **measured here, not applied** (it touches
`act_scales2_f32.bin`).

## Result 3 — the conv's OUTPUT grid is worth exactly nothing: bit-identical final codes

`--convout` keeps the conv's exact requant value in float32 instead of rounding it to int8 at
`SA_out` (taps, per-tap int8 rounding, clamps and the input build all unchanged). Result:

| arm | final PSNR (2.0) | cosine | final codes |
|---|---|---|---|
| `wire` | 20.587570230809145 dB | 0.8885014653205872 | — |
| **`convoutfp`** | **20.587570230809145 dB** | **0.8885014653205872** | **bit-identical** |

Per-layer cosines differ in the 11th digit and the walk's clamp count drops 553,689 → 452,085, yet
the decode's final int8 codes are **identical to the digit**. The conv's output rounding is
absorbed by the next op's requantisation at every layer — the norm that follows re-renders the
tensor on a grid ~15× finer *relative to its own values*, so sub-LSB detail at `SA_out` never
reaches the end product. Together with P3.15 (`noround` +0.04 dB) this closes the whole rounding
class: **nothing in the walk's quantisation chain explains the 47 % rel-RMSE.**

## Result 4 (P3.19) — the input-build probe, and the confound it cannot yet remove

`tools/npu/vae_gather_probe.py` rebuilds each conv's input from the **previous conv's reference
codes** with the program's own wiring and scores it against the reference's own conv input, so
upstream drift is removed by construction. Only three conv→conv adjacencies exist in the wire.

| src → dst | module | wiring | frames | cosine | codes differ | |Δ| (codes) |
|---|---|---|---|---|---|---|
| 11 → 12 | `up_blocks.0.upsamplers.0.resample.1` | 2 | 3 → 5 | 0.971035 | 93.84 % | 6.47 |
| 12 → 13 | `up_blocks.1.resnets.0.conv_shortcut` | 0 | 5 → 5 | **1.000000** | 0.00 % | 0.00 |
| 20 → 21 | `up_blocks.1.upsamplers.0.resample.1` | 2 | 5 → 9 | 0.976613 | 51.95 % | 3.46 |

The pass-through row is **exact**, which validates the method. The two shuffle rows are **not yet
evidence against the wiring**: they are confounded by the known `time_conv` streaming-vs-whole-
sequence gap, because P3.17 replaced `tr[11]`/`tr[20]` with the whole-sequence rebuild, and that is
the tensor this probe feeds the gather. The signature is exactly a frame-0 effect and nothing else:

| site | per-frame cosine | per-frame mean |Δ| |
|---|---|---|
| 11 → 12 | 0.9045 · 0.9887 · 0.9852 · 0.9930 · 0.9885 | 14.07 · 4.89 · 4.82 · 4.02 · 4.53 |
| 20 → 21 | 0.8881 · 0.9764 · 0.9805 · 0.9866 · 0.9866 · **1.0000 · 1.0000 · 1.0000 · 1.0000** | 13.58 · 5.22 · 4.60 · 3.92 · 3.80 · **0 · 0 · 0 · 0** |

Four of nine frames are **bit-exact** through the shuffle *and* the 2× upsample *and* the requant;
the loss is concentrated at frame 0 and decays over the first frames. So the shuffle arithmetic is
right where it can be checked clean, and the residual tracks the streaming `time_conv` difference
— which stays "reported, not applied" until the streaming form is fed to the walk deliberately.

## What this changes

The paper's decomposition cannot say "13.75 dB of glue". Measured: glue hops 0.013 dB, conv output
rounding 0.000 dB, conv input build 0.0004 dB, per-tap scheme 0.04 dB (P3.15), conv intrinsic
0.14 pt of cosine (P3.17). **Every quantisation hop the program performs is individually and
jointly worth ≤ 0.06 dB**, so the decode's 47 % rel-RMSE is carried by how the walk's *state*
differs from the reference's — the accumulation itself, or one of the two structural items this
project already has open (the streaming `time_conv` form, and the third upsample's post-ADD
input at layer 28, which has no conv→conv adjacency to probe).

## Next actions, in order

1. **The restart census** — restart the walk from the reference's own state at three block
   boundaries (after conv 11 / 20 / 27) and score the end product. That localises **where** the
   drift is generated instead of testing one op class at a time; 3 host-only arms, ~12 min,
   no device.
2. **The streaming `time_conv` arm** — store the streaming tensors alongside the rebuild in
   `reference_trace` and re-run the gather probe, so the two shuffle rows stop carrying the
   confound; then the wiring is either convicted or cleared.
3. **`post_quant_conv` (+7.32 dB, P3.15) and the conv-3 norm's 0.97 % clamp** — both are measured
   and both edit `artifacts/vae`/`act_scales2_f32.bin`, so both stay **the student's call**.
4. Then `proc_b --decode 1` (`convwire`) → Gate 5, unchanged.
