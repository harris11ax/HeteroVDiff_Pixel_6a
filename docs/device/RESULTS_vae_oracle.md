# P3.17 — inherited or intrinsic? The same convs on the reference's own input

*2026-11-16 10:06. Host-only: **no device run, no device rate touched** — `T_V` 8.908 s,
`T_N` exact 0.8905 s/block, `k_b*` 165 are exactly as committed. Instrument
`tools/npu/vae_sim.py --oracle` (new arm + a measured correction to the reference trace),
capture `docs/device/vae_sim_20261116_100611.{json,md}`, latent
`artifacts/vae/dump_20261116_080721` (the post-fix device walk's own latent).*

## What was unrun, and what it cost not to know

P3.16 closed both static-scheme candidates at layer 34 and named the last question its
instrument could not ask: is layer 34 **intrinsically** bad (its weights / output scale / tap
scheme), or is it only where **inherited** error lands? The discriminating probe is one arm —
feed each conv the reference's own input, quantised at the program's `SA_in`, and score its
output. Same weights, same per-tap bias, same per-tap rounding, same output scale; the only
thing removed is the error the walk had already accumulated.

## Result 1 — the validator holds, and the answer is unanimous

| arm | final PSNR (2.0) | cosine | rel-RMSE | worst layer |
|---|---|---|---|---|
| **`wire`** | **20.5876 dB** | 0.888501 | 47.12 % | 34 (0.714915) |
| **`oracle`** | **34.34 dB** | 0.995595 | 9.68 % | 35 (0.995406) |

`wire` reproduces P3.15/P3.16 to the digit (20.5876 / 0.888501 / 0.714915), so the two arms
are directly comparable and comparable to the phone (P3.15 measured the simulator tracking
the device to 0.15 dB).

**Every one of the 36 convs is faithful given a good input.** Mean intrinsic loss
**0.143 pt of cosine**, median 0.09, **max 0.46** (at `conv_out`); **0/36 sites lose more
than 1 pt** on a perfect input. Against that, the mean *inherited* loss is **5.01 pt**, and
in the last up_block it reaches **15.18 pt**.

| up_block (layers) | mean inherited | mean intrinsic |
|---|---|---|
| mid_block + up_blocks.0 (0–11) | 1.47 pt | 0.14 pt |
| up_blocks.1 (12–21) | 4.35 pt | 0.09 pt |
| up_blocks.2 (22–28) | 6.00 pt | 0.13 pt |
| **up_blocks.3 (29–35)** | **11.03 pt** | **0.24 pt** |

## Result 2 — layer 34 is exonerated

`decoder.up_blocks.3.resnets.2.conv2`: input **0.848233**, wire output **0.714915**, oracle
output **0.999143** ⇒ **intrinsic 0.09 pt, inherited 15.18 pt**. A 3×3×3 conv on the
reference's own input reproduces that module to 0.9991; the 28.5-pt drop P3.15/P3.16 kept
ranking is entirely what it was handed. The named defect site is not a defect site, and the
two candidates P3.16 priced (add grid +0.0016 dB, `conv_out` per-channel +0.0015 dB) were
already consistent with this — they moved nothing because there was nothing at the site to
move.

## Result 3 — a hole in every table to date: layers 11 and 20 were never scored

The oracle arm's shape guard fired at layer 11 and refused to substitute, and the reason is a
measured structural fact, not a bug in the new arm:

`AutoencoderKLWan._decode` runs the decoder **one latent frame at a time** with a causal
`feat_cache`. `WanResample.forward` (`upsample3d`) **skips `time_conv` for the first latent
frame** — it sets a `"Rep"` sentinel and emits that frame's two halves by *replicating the
block's input*, with no conv at all — then calls `time_conv` one frame at a time with a
zeroed causal cache. Measured consequence: the module's hooked input and output are
**T−1 frames** (2 of 3 at layer 11, 4 of 5 at layer 20), so `tr[name]` never matched the
program's `[T, 2C, H, W]` tensor and **every arm from P3.14 to P3.16 printed
`SHAPE MISMATCH` for these two layers**. 2 of 36 convs were absent from all cosines.

It also explains P3.14's asymmetric gather mapping: the reference's frame 0 is `b[0]` half 0
**because it is the conv's input, not a conv of it**.

Reported, not "fixed": the walk's whole-sequence `time_conv` and the streaming reference
differ **in kind** at frame 0 (convolution vs replication), which is the open item already
recorded as "the layer-11/20 3-frame-vs-2-frame `time_conv` support". What the tool now does
is make the two layers *scorable*: it hooks the upsample **block**'s input (the conv's real
input, all T frames) and rebuilds the target by running the same module in whole-sequence
form on it. Logged per run, and confined to those two layers — layer 12's cosine is
unchanged (0.963801, to the digit, vs P3.16).

| layer | module | streaming in/out | whole-sequence | `wire` cos | `oracle` cos |
|---|---|---|---|---|---|
| 11 | `up_blocks.0.upsamplers.0.time_conv` | 2 / 2 frames | 3 / 3 | 0.986019 | 0.996160 |
| 20 | `up_blocks.1.upsamplers.0.time_conv` | 4 / 4 frames | 5 / 5 | 0.961396 | 0.997462 |

**36/36 convs now carry a measured cosine for the first time.**

## Result 4 — the convs are not the lever; the glue is

The per-layer table (`vae_sim_20261116_100611.md`) shows where the loss is made. Between
layers 29 and 35 the *inherited* column climbs **7.02 pt (layer 29) → 15.18 pt (layer 34)**
while every site's own rounding stays ≤ 0.46 pt, and the sum of all conv-side rounding up to
layer 34 is only **4.7 pt**. So the ~18 dB gap is produced by the ops this arm does not
replace — the
`WanRMS_norm`+SiLU requant, the residual add on `post_add_sa`, and the gather requant — and
it is *amplified* as it grows (the last up_block loses 2–3× what the earlier ones do for the
same per-site cost).

That is the same conclusion P3.16 reached from the static arms (0.003 dB between them), now
carrying a per-layer number: **no individual conv or scale is at fault, so no per-layer
quantisation tweak can recover the decode.** The levers that remain are the ones already
priced — `post_quant_conv` **+7.32 dB** (P3.15, edits the artifact ⇒ the student's call) —
and, newly named, the **glue's own one-hop cost**, which this instrument cannot yet measure
because it compares convs only.

## Not claimed

- **No device number.** `T_V` = 8.908 s, `T_N` = 0.8905 s/block, `k_b*` = 165 all stand.
- **34.34 dB is not a reachable program.** It is the bound under "every conv is handed the
  reference's own input" — a probe, not a proposal: it needs a perfect decoder at every layer
  to build its own inputs.
- **No mechanism** for the amplification in the last up_block beyond the measurement above;
  the glue ops are not individually instrumented yet.
- The streaming-vs-whole-sequence `time_conv` difference at frame 0 is **reported, not
  applied** — changing the walk there edits the artifact's program, and P3.14's gather mapping
  was measured *around* it.

## Method notes

- `reference_trace` now pre-hooks **every** conv's input (not just `resample.1`'s), quantising
  inside the hook: 36 convs of up to 113 M elements would otherwise cost ~5.6 GB of float32
  clones for inputs alone.
- Arm cost: fp32 reference 34 s + `wire` walk 3 min 30 s + `oracle` walk 3 min 30 s ≈ **7 min
  in one process**, so it is launched detached with `docs/device/.measurement_in_flight`
  written first (cron tick limit is inactivity-based).
- The `oracle` arm's `cos_gather` reads 1.000000 at every layer by construction — a free
  assertion that the input file round-trips the exact codes the conv consumed.
- One bug paid for before any number was produced: the input hook was defined once with a
  late-binding `n` (NameError at definition); it is now a factory per module.
