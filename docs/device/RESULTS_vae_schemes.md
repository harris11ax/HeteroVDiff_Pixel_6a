# P3.15 — the accumulating scheme is priced, and it is *not* the missing 18 dB

*2026-11-16 08:33. Host-only, **no device run and no device rate touched**: `T_V`, `T_N`,
`T_G` and `k_b*` are exactly as they were. Instrument `tools/npu/vae_sim.py` (extended:
`--schemes`, `--pqc`), capture `docs/device/vae_sim_schemes_20261116_083315.{json,md}`,
latent `artifacts/vae/dump_20261116_080721` (the post-fix device walk's own latent).*

## What was unrun

P3.14 ended with 20.44 dB and named three accumulating-scheme candidates for the other
~18 dB: **(1)** the per-tap INT32 bias counted `kT` times (P3.13's measured defect),
**(2)** each tap rounded/clamped to int8 before the sum, **(3)** the missing
`post_quant_conv` (16→16). It proposed scoring them with the simulator, since that costs
no phone time. This does that — both as a full 2×2 over the first two, and with (3)
measured against a target that *has* the op in it.

## Method — four arms, one factorial

`vae_sim.py --schemes all --pqc` walks `vae_wire.txt` with the measured `shift` gather
(P3.14's fix) and the device's own semantics, changing one thing at a time:

| arm | INT32 bias | tap accumulation |
|---|---|---|
| `wire` | on **every** tap (as `npu_block.cpp` builds it) | each tap requantised+clamped to int8, summed, clamped |
| `bias1` | on the **first applied** tap only | same |
| `noround` | on every tap | summed in int32, requantised **once** |
| `exact` | first applied tap only | summed in int32, requantised once |

`bias1` = candidate 1 alone, `noround` = candidate 2 alone, `exact` = both. For frame 0
of a causal conv, tap 0 has no source frame, so "first applied" is the device's own
"tap 0's graph carries the bias" behaviour and the number of frames where that differs
is recorded per layer (`tap0_skipped`).

**Validator first:** `wire` must reproduce the device. It scores **20.59 dB / cosine
0.8885**, against the device walk's measured **20.44 dB / 0.8967** (`vae_psnr_fix_…_080912`)
— 0.15 dB, i.e. the simulator tracks the phone, so the arms below are comparable to the
device number and not just to each other.

## Result 1 — the end product, per arm

| arm | final PSNR (2.0) | cosine | rel-RMSE | worst layer | max clamp % |
|---|---|---|---|---|---|
| **`wire`** (what the phone runs) | **20.59 dB** | 0.888501 | 47.12 % | 34 (0.7149) | 8.54 |
| `bias1` (bias once) | 20.30 dB | 0.873801 | 48.70 % | 34 (0.6325) | 8.18 |
| `noround` (requant once) | **20.63 dB** | 0.889650 | 46.89 % | 34 (0.7159) | 6.11 |
| `exact` (both) | 20.32 dB | 0.874252 | 48.62 % | 34 (0.6319) | 5.91 |
| `exact+pqc` (both + the missing op) | 19.65 dB | 0.839626 | 54.51 % | 34 (0.6032) | **0.43** |

*(`exact+pqc` is scored against a different target — see Result 3 — so its row is not
rankable against the four above.)*

**Candidate 2 is worth +0.04 dB** (20.59 → 20.63, cosine +0.0011): free, and inside the
noise of everything else on this page. **Candidate 1 is worth −0.29 dB**: removing the
bias over-count makes the decode *worse*, not better. Doing both is −0.27 dB.

## Result 2 — the two metrics disagree, and the disagreement is localized

Per layer, the bias arms are better nearly everywhere:

| arm | mean cosine, layers 0–29 | mean cosine, layers 30–35 | layers ≥ 0.99 |
|---|---|---|---|
| `wire` | 0.963007 | 0.829615 | 4 / 34 |
| `bias1` | 0.972510 (**+0.0095**) | 0.833091 (+0.0035) | 11 / 34 |
| `noround` | 0.963608 (+0.0006) | 0.830763 (+0.0011) | 4 / 34 |
| `exact` | 0.972882 (**+0.0099**) | 0.833382 (+0.0038) | 11 / 34 |

`conv_in` moves 0.9959 → **0.99995**, layers 1–10 all clear 0.995, and the first
divergence moves from **layer 3 to layer 12** — the bias over-count is a real per-tap
defect, exactly as P3.13 measured it. Yet the **final** decode is 0.27–0.29 dB *worse*,
and the reason is visible in one row: at **layer 34** (`decoder.up_blocks.3.resnets.2.conv2`)
`wire` scores 0.7149 and both bias arms collapse to **0.632**. The over-count is
partially cancelling an error at the last up_block. So the two candidates cannot be
priced independently of layer 34 — fixing the bias now would trade +0.0099 mean layer
cosine for −0.29 dB of end product.

## Result 3 — the missing `post_quant_conv` is worth +7.32 dB

The op is not in the artifact (36 layers = decoder only) and the program has no op for
it, so it was reported and not folded. Scored, not adopted: the reference is re-run
**with** the op (attention still skipped, as the program skips it), the walk is fed the
op's quantised output (`sa = 0.0273799` from its own max, **0 clamped codes**), and
layer 0's input scale is re-calibrated to that value. Both variants are then measured
against the **same** target.

| variant | PSNR (2.0) | cosine | rel-RMSE |
|---|---|---|---|
| `exact` — the program as it exists (no `post_quant_conv`) | 12.33 dB | 0.138559 | 126.52 % |
| `exact+pqc` — with the op folded in | **19.65 dB** | 0.839626 | 54.51 % |

**+7.32 dB from one 16→16 conv**, and it drops the walk's worst clamping from 5.91 % to
**0.43 %** (the op's output is properly scaled where the latent was not). For scale:
the error it removes is 25× larger than candidates 1 and 2 combined. It is not a small
16×16 matrix either — fp32 vs fp32, op on vs bypassed, is **12.06 dB / cosine 0.179**,
i.e. the op changes the decode by as much as the whole remaining deficit.

Caveat that belongs with the number: this branch **edits the artifact** (a 36th-layer
conv, 16→16, needs its own INT8 weights and scales; the existing artifact is
`72,685,152 B` INT8 unpadded), so it is **the student's call, measured not applied**, and
its layer-0 input scale here is a one-sample calibration (`max|x|/127` on this tensor),
not a `calibrate.py` sweep.

## What this changes

- **The accumulating scheme is closed as an explanation.** Candidates 1 and 2 are
  ±0.3 dB against a 20.59 dB decode; the ~18 dB is elsewhere. P3.14's ranking
  ("per-tap bias ~1 % per layer is the next candidate") is **refuted at the end-product
  level** — it is 1 % *per layer* and negative cumulatively.
- **The missing `post_quant_conv` is the largest measured item left** and the only one
  that needs a decision rather than a code change.
- **Layer 34 is now the named defect site**: worst layer in all five arms, and the two
  late layers (34, 35) are where the bias arms lose their end-to-end advantage. That is
  cheap to attack with this instrument — `up_blocks.3.resnets.2` is the last resnet, so
  the candidates are its residual add at `post_add_sa` (P3.11 measured post-add range up
  to 4.76× conv2's own), and `conv_out`'s 96→3 shape.

## Not claimed

No mechanism for the layer-34 cancellation. No device run, so **no rate is re-derived**:
`T_V` = 8.908 s (8.8226 conv + 0.0855 Mali attn), `T_N` exact 0.8905 s/block,
`k_b*` = 165 all stand as committed. Nothing locked was touched.

## Method notes

- `vae_sim.py` grew `--schemes {all,wire,bias1,noround,exact}` and `--pqc`; every arm
  records its own `bias`/`round` mode, its final PSNR, and `tap0_skipped` per layer, so a
  later rerun cannot silently compare two different schemes.
- The pqc branch needed `sain_override` (the per-`ai` input scale), because folding an
  op in front of layer 0 changes the tensor layer 0 consumes; leaving `SA_in[0]`
  untouched would have quantised the new latent at the old tensor's scale.
- `final_metrics` reuses `vae_psnr.py`'s metric block verbatim (data_range 2.0) so a
  simulator number and a device number are the same quantity.
- Runtime: 4 walks + 1 pqc walk + 2 fp32 references = **15 min 20 s** host, one process.
  A walk is ~3 min; a fp32 reference is ~34 s. No harness epilogue cost — that is a
  device-side term and does not exist here.
