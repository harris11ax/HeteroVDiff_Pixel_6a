# P3.13 — the device decode gets a quality number, and it does not decode yet: two defects found, one fixed

*2026-11-16 07:36. Device runs `vae_wire_dumpr_20261116_073415` (full walk, dump),
`vae_dump_layer0_fix_20261116_073615` (layer-0 probe); host analysis
`vae_psnr_20261116_073602`, `conv_parity_20261116_073621`. Serial `25311JEGR11811`,
same boot. **Mechanism-grade on thermal** (peak 24.0 °C vs `battery+3` 23.9 → 0.1 °C
over; the walk was 44.9 s and ended at 39 °C). No device **rate** is quoted here —
`T_V`/`T_N`/`T_G`/`k_b*` are untouched.*

## What was unrun

P3.12 could only say the wired walk *executed*: 592/592 `CONV_2D`, saturation 0.0721 %.
Nothing compared its output to anything. `convwire` had no way to emit its own result,
so "the decode decodes" was untested by construction.

## The instrument

- `npu_block convwire <convs> <pc> 1 [dump_layer]` now writes, on the device, the final
  activation codes **and** the latent codes the first conv actually consumed, each with
  its own declared scale (`/data/local/tmp/vae_dump.bin|.txt`, `vae_latent.bin|.txt`).
  With `dump_layer = L` it instead writes that conv's exact int8 **input and output** and
  stops — one layer, both ends, nothing assumed.
- `tools/npu/vae_psnr.py` dequantises both and scores the device's decode against the
  fp32 reference **from the identical quantised latent**, in three references:
  A fp32/attention-on, B fp32/attention-skipped (the program the device runs),
  C INT8 weights + INT8 activations at the measured `SA` (the same quantisation
  simulated on the host).
- `tools/npu/conv_parity.py` scores one layer against an exact emulation of its taps,
  with the pairing between tap index and frame **measured** against the checkpoint.

Guards, all of which earned their keep: the dumped latent is re-derived from the device's
own LCG (bit-identical, 97,920 codes) and its scale is asserted equal to the table's
`SA_in[0]` (0.0198558513, delta 0) — so the host and the device quantise the same tensor.

## Result 1 — the number, and it is a failure

| reference (same latent) | rel-RMSE | cosine | PSNR (2.0) | SSIM |
|---|---|---|---|---|
| A fp32, attention ON | 129.31 % | 0.2190 | 11.60 dB | 0.054 |
| B fp32, attention skipped | 131.82 % | 0.2103 | 11.65 dB | 0.054 |
| C INT8 weights + acts at `SA` | 132.71 % | 0.2081 | 11.68 dB | 0.055 |

The device's decode is **uncorrelated with its own quantisation simulated on the host**
(C), so this is not a scheme cost — it is a defect in the walk. Gate 1's 38.73 dB
simulation stands as a simulation.

## Result 2 — defect found and FIXED: the tap→frame labels were mirrored

`WanCausalConv3d` pads **2 zeros at the front** of the time axis, so weight slice 0 is
applied to `x[f-2]`; `vae_export` labels slice 0 `"f"`. Emulating both pairings on
`decoder.conv_in` against the checkpoint module:

| pairing | rel-RMSE vs checkpoint | cosine |
|---|---|---|
| as labelled (`tap t ↔ f-t`) | 113.48 % | **0.0558** |
| mirrored (`tap t ↔ f-(kT-1-t)`) | 0.764 % | **1.00005** |

Every 3-tap conv in the walk had been reading the wrong frames — 36 layers, 100 taps.
Fixed in `tools/npu/vae_wire.py` (`d = (ntaps-1) - d_label`, with the measurement in the
comment). The fix is visible in the walk's own counters: overall clamp **0.0841 % →
0.0149 %**, and `conv_in` clamp 1.57 % → 4.45 % (its output is now the conv it claims).
End-to-end did **not** improve (cosine 0.28 → 0.21), which is what located defect 2.

## Result 3 — the HAL is NOT the problem on `CONV_2D` (this shape)

Layer 0, device codes vs the exact emulation from the same dumped input codes:

| emulation | bit-exact codes | within ±1 | mean abs Δcode | cosine |
|---|---|---|---|---|
| **device semantics** (per-tap int8, bias per tap) | **98.59 %** | **100.00 %** | 0.01 | **1.000312** |
| correct semantics (one accumulation, bias once) | 9.13 % | 12.96 % | 7.20 | 0.990842 |

So the Edge TPU computes exactly what the program asks (98.59 % of codes bit-identical,
100 % within one code) — unlike `FULLY_CONNECTED` in Gate 3. The program itself is ~1 %
off per layer because of two real but small defects, now quantified: each temporal tap's
graph carries the **same INT32 bias** (bias counted `kT` times) and each tap is rounded to
int8 before the sum.

**Therefore the remaining decode error is in the glue between convs** — `NORMSILU`,
`ADD`/`SAVE`/`LOAD`, or the temporal-shuffle / 2× upsample gathers — not in the conv and
not in the HAL. That is the next measurement, and it is host-side first.

## Also recorded: `post_quant_conv` is missing from the program

`vae.decode` applies `post_quant_conv` (16→16, non-identity, absmax 0.250) **before**
`decoder.conv_in`; it is not in the artifact (36 layers = decoder only) and the program has
no op for it, so the device's `conv_in` consumes the latent directly. The host reference
therefore bypasses it — otherwise the comparison is between two different inputs, which is
exactly what a first attempt produced (0.46-cosine "failure"). Reported as a deviation, not
folded into the artifact: folding it into `conv_in`'s taps would be exact but would edit a
verified artifact.

## Method defects paid for en route

- **`[f][h][w][c]` is not `[B][C][F][H][W]`.** The harness stores channel-last with frames
  outermost; feeding that to `vae.decode` without moving the frame axis inside the channel
  axis raised `expected 16 channels, got 3`.
- **A single-conv dump taken with `--convs 1` is post-norm, not post-conv** — the loop
  breaks at the *next* conv, after the intervening ops ran. That is why `--dump-layer`
  exists (it dumps inside the conv and returns).
- **`--dump` is 3,525,120 B of evidence**; a stale dump directory is indistinguishable from
  a fresh one, so every capture records its own `dump` block with per-file byte counts.

## Order of next actions

1. **Localize the glue.** Host-side program simulator (`vae_wire.txt` + artifact) that
   reproduces the walk, then compare layer-by-layer against per-layer device dumps at
   3–4 probe layers (cheap: `--dump-layer L`). Candidates already suspicious: the
   temporal-shuffle gather (`want_f = 2f-1`, `half = t%2`), the `ADD` at `post_add_sa`, and
   the `SAVE`/`LOAD` slot semantics.
2. Fold `post_quant_conv` into the program (a 16→16 1×1 conv, or fold into `conv_in` —
   reported, needs the student's call because it edits a verified artifact).
3. Per-tap bias: give tap 0 the bias and the others zero (one-line program change), which
   recovers the ~1 %-per-layer program error measured above.
4. Only then `proc_b --decode 1` → `convwire`, and Gate 5.
