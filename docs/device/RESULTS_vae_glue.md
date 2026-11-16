# P3.14 — the decode's glue: the temporal shuffle was wrong, and fixing it is worth +8.8 dB

*2026-11-16 08:14. Device run `vae_wire_fix_20261116_080721` (full walk, dump), host
analysis `vae_psnr_fix_20261116_080912`, simulator runs `vae_sim_shuffle_20261116_080227`
/ `…_080342` / `…_080615`. Serial `25311JEGR11811`, same boot. **Mechanism-grade on
thermal** (peak TPU 24.0 °C vs `battery+3` 23.6 → 0.4 °C over; walk 44.9 s, ended 39.0 °C).
No device **rate** is claimed or changed here — `T_V`/`T_N`/`T_G`/`k_b*` are untouched.*

## What was unrun

P3.13 ended with: *"the remaining decode error is in the glue between convs —
`NORMSILU`, `ADD`/`SAVE`/`LOAD`, or the temporal-shuffle / 2× upsample gathers"* — and
the instrument that would localize it was a host-side program simulator. That is
`tools/npu/vae_sim.py`, and it found the defect on its first walk.

## The instrument

`vae_sim.py` walks `docs/device/vae_wire.txt` op for op **with the device's own
semantics**: int8 codes at the declared per-tensor scales, the per-tap split with each
tap rounded to int8 before the sum, the INT32 bias in accumulator units
(`bias/(sa_in·sc)`), the same `g = act.sa/sa_in` requant in the wiring step, the same
round-half-away-from-zero. It reads the **dumped latent codes** the device consumed, so
both sides quantise the same tensor.

It is diffed against the fp32 reference's **own intermediate tensors**: the reference
decode (attention skipped, `post_quant_conv` bypassed — the program's deviations) is run
with a forward hook per decoder module, the hook's calls concatenated along the frame
axis (the decoder is iterative; the chunks concatenate to exactly the plan's 3 / 5 / 9
frames — checked, not assumed), each quantised at that layer's own `SA_out`.

One conv per layer, so the first cosine collapse names the op range between conv L−1
and conv L.

## Result 1 — the glue breaks at exactly one place, and it is the shuffle

| layer | module | cosine vs fp32 (wire gather) |
|---|---|---|
| 0 | `decoder.conv_in` | 0.9959 |
| 1–4 | `mid_block.resnets.{0,1}.conv{1,2}` | 0.9937 / 0.9951 / 0.9739 / 0.9868 |
| 5–10 | `up_blocks.0.resnets.{0,1,2}.conv{1,2}` | 0.9827 … 0.9646 |
| 11 | `up_blocks.0.upsamplers.0.time_conv` | shape mismatch (ref 2 frames, program 3) |
| **12** | **`up_blocks.0.upsamplers.0.resample.1`** | **0.7172** |

Ten convs of norms, SiLUs and residual adds track the reference at **0.965–0.996**, so
`NORMSILU` and `ADD` are *not* the defect. The collapse is the first temporal upsample.

## Result 2 — the mapping, measured off the reference rather than read off the model

The wire builds each resample conv's input by gathering buffer frame `t/2` and channel
half `t%2` — both halves of `b[0]`, then both of `b[1]`, …, dropping the last half of
`b[T−1]`. Correlating the reference's **own `resample.1` conv input** (5 frames, captured
with a pre-hook) against every candidate gives the truth:

| ref frame | x0 | b0·h0 | b0·h1 | b1·h0 | b1·h1 | b2·h0 | b2·h1 |
|---|---|---|---|---|---|---|---|
| 0 | 0.392 | **0.896** | 0.891 | 0.251 | 0.283 | 0.248 | 0.267 |
| 1 | 0.274 | 0.265 | 0.255 | **0.977** | 0.810 | 0.260 | 0.270 |
| 2 | 0.282 | 0.257 | 0.301 | 0.821 | **0.972** | 0.251 | 0.332 |
| 3 | 0.277 | 0.280 | 0.268 | 0.287 | 0.268 | **0.970** | 0.787 |
| 4 | 0.286 | 0.271 | 0.324 | 0.296 | 0.367 | 0.796 | **0.963** |

So `WanResample.forward` emits **1 + 2(T−1) = 2T−1** frames as

    frame 0        = b[0], channel half 0 only
    frame 2j−1,2j  = b[j] halves 0 and 1        (j = 1 … T−1)

The wire emitted both halves of `b[0]` and then dropped `b[T−1]`'s last half. **The frame
COUNT is identical either way (2T−1), which is why it failed silently** — no shape error,
no clamp alarm, and the saturation even *looked* better (P3.12: 0.0721 % worst layer).

## Result 3 — the fix, and what it is worth

`npu_block.cpp`, the wiring step: `(t/2, t%2)` → `t==0 ? (0,0) : ((t+1)/2, (t+1)%2)`.

| measurement | before | after |
|---|---|---|
| resample conv **input** vs the reference's own conv input (5 frames) | 0.6114 | **0.9536** |
| layer 12 conv **output** vs fp32 | 0.7172 | **0.9638** |
| **device decode vs fp32 from the same latent** (A, attention ON) | 11.60 dB / cos 0.2190 | **20.44 dB / cos 0.8967** |
| device vs B (attention skipped) | 11.65 dB / 0.2103 | 20.40 dB / 0.8946 |
| device vs C (INT8 both operands simulated) | 11.68 dB / 0.2081 | 20.42 dB / 0.8948 |

+8.84 dB for a one-line index change. The device run itself is unchanged in every other
respect: **36/36 convs, 592/592 `CONV_2D` executes, `pc=1` on all 36, 0 fallbacks, compile
18.81 s, wall 44.932 s**, and the saturation **falls again** (0.0149 % → 0.0137 %
overall). `conv_ms` 12,274.3 ms against the same-binary previous walk's 12,318.9 ms.

## Result 4 — the second temporal upsample, scored: the same fix carries, and no further collapse

`--shuffle shift --max-layer 21` (capture `vae_sim_l21_20261116_081109`) walks past the
second upsample with the same gather, and the reference input/output now agree:

| layer | module | cosine vs fp32 |
|---|---|---|
| 13 | `up_blocks.1.resnets.0.conv_shortcut` | 0.9680 |
| 14–19 | `up_blocks.1.resnets.{0,1,2}.conv{1,2}` | 0.9391 … 0.9664 |
| 20 | `up_blocks.1.upsamplers.0.time_conv` | (2-frame vs 3-frame shape, as layer 11) |
| **21** | **`up_blocks.1.upsamplers.0.resample.1`** | **0.9488** (gather 0.9445) |

So there is **no second structural defect between layers 12 and 21**: the error now drifts
gradually (0.996 → 0.94) instead of collapsing at one op, which is what an accumulating
scheme cost looks like — per-tap bias counted `kT` times (~1 % cosine per layer, P3.13),
per-tap int8 rounding before the sum, and the missing `post_quant_conv`. Those are the
next candidates, and the simulator prices each one without touching the phone.

## What is still wrong

20.44 dB is not a decode: Gate 1's *simulation* of the same quantisation is 38.73 dB, so
~18 dB of error remains **inside the walk**, not in the scheme. The same instrument now
points at the rest of it:

1. **The second temporal upsample** (layer 21, `up_blocks.1.upsamplers.0.resample.1`) has
   the same gather and is now fixed by the same change, but it has not been *scored* —
   `--max-layer 21` does that in a minute.
2. **Layer 11's own shape** — the reference's `time_conv` sees 2 frames where the program
   gives it 3 (frames 1 and 2 are 2-tap, frame 0 is 1-tap in the streaming decode; the
   whole-tensor conv gives frame 2 three taps). If that residual matters it is the next
   candidate, and the simulator can price it before any device run.
3. `post_quant_conv` is still absent (reported, not folded) and per-tap bias is still
   counted `kT` times.

## Method defects paid for en route

- **A 4-dim hook output is a `Conv2d`'s `[b·t, c, h, w]`, not a video tensor.** The
  resample convs receive the frames folded into the batch axis; concatenating them on
  dim 0 is the frame order, and the total must equal the plan's frame count — assert it,
  do not assume the layout.
- **`np.rint` is half-to-even; NNAPI rounds half away from zero.** `q_clamp` implements
  the latter, so an exactness claim about the device is not silently a claim about numpy.
- **Exact int accumulation in float32 needs K ≤ ~1039 terms** (`127·127·K < 2^24`), so the
  conv's `Cin` is chunked (96 channels → K = 864) and the partials are summed in int32.
  Without that, a "the emulation is exact" claim is false by a few LSB — and the P3.13
  result it is compared against is a *bit-exactness* number.
- The first hypothesis (frame 0 = the pre-`time_conv` activation, then 2-frame windows)
  made the cosine **worse** (0.717 → 0.620). It was cheap to falsify because the
  simulator costs no phone time — that is the whole reason to build one.
