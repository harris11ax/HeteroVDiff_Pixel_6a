# Gate 1 — INT8 artifact integrity (Phase 1 deliverable 1)

*Measured 2026-11-15 20:5x, host only. Reproduce with
`python phase1_quant/quantize_int8.py && python phase1_quant/verify_artifact.py`.
No device number appears in this document.*

The per-output-channel INT8 base weights now exist as a packed artifact, and an **independent** pass
re-derives them from the FP32 checkpoint. The quantiser's own report is not the evidence:
`phase1_quant/verify_artifact.py` re-reads the bytes on disk, re-quantises the source, and demands
byte identity.

## What is on disk

| Path | Content | Bytes |
|---|---|---|
| `artifacts/blocks/block_{00..29}.bin` | INT8 base GEMM weights of one DiT block, packed in manifest order, each tensor 4096-B aligned | 46,399,488 each |
| `artifacts/blocks/block_{L}.scales.f32` | one FP32 scale per output channel, same order (22,784 scales) | 91,136 each |
| `artifacts/blocks/block_{L}.fp16.bin` | the block's FP16 remainder: 10 biases, 4 attn norms, `norm2`, `scale_shift_table` [1,6,1536] | 100,352 each |
| `artifacts/top.fp16.bin` | FP16 patch embedding [1536,16,1,2,2], `condition_embedder` linears/biases, `proj_out`, top `scale_shift_table` [1,2,1536] (15 tensors) | 51,562,496 |
| `artifacts/manifest.json` | every tensor's offset, shape, scale offset, per-file sha256, Gate 1 verdict fields | 30 blocks / 300 GEMMs |

Region order inside a block (`manifest.json:gemm_order` — the device must use it):
`attn1.{to_q,to_k,to_v,to_out}` → `attn2.{to_q,to_k,to_v,to_out}` → `ffn.net.0.proj` (up,
`[8960,1536]`) → `ffn.net.2` (down, `[1536,8960]`). Row-major `[N_out, K]`, `K` contiguous, exactly
the layout Gate 2 verified for both `char4`+`arm_dot` and NNAPI `FULLY_CONNECTED`. No padding is
needed: every base GEMM's byte count is already a multiple of 4096.

`artifacts/` is gitignored — 1.4 GiB of regenerable bytes. The manifest regenerates with them
(40.1 s quantise, 61 s verify, both deterministic: the second run reproduced every byte).

## Gate 1 bars

| Bar (`docs/05_phase_plans.md` §Phase 1) | Required | Measured | Verdict |
|---|---|---|---|
| total INT8 bytes | 1,391,984,640 ± 1 MiB (1.296 GiB) | **1,391,984,640 B (delta 0)** | PASS |
| per-region size | ≤ 47.29 MiB and ≤ AHB ceiling (1792 MiB) | **44.25 MiB** | PASS |
| per-GEMM cosine vs W | > 0.999 | **0.999922 worst of 300** | PASS |
| per-GEMM rel. Frobenius | ≈ 3–5 % expected | **0.93 % MAC-weighted, 1.25 % worst** | better than expected |
| end-to-end FP32-vs-INT8 PSNR ≥ 28 dB / SSIM ≥ 0.92 | before any device work | **not run — blocked, see below** | TBD |
| coverage | every checkpoint tensor accounted for | **1317 = 300 INT8 + 525 FP16 + 492 expert deferred, 0 in no file** | PASS |

Byte identity: all 300 GEMMs' INT8 payloads and all 683,520 scales equal a fresh quantisation of the
checkpoint bit for bit; every file's sha256 matches the manifest.

## Numbers

- **300 GEMM instances** (30 blocks × 10 GEMMs), **1,391,984,640 INT8 bytes**, 683,520 scales.
- **Weight reconstruction**: MAC-weighted relative Frobenius error **0.93 %**, cosine **0.999956**.
  Range over all 300: **0.66 % – 1.25 %**. Worst instance: block 3 `ffn.net.2` `[1536,8960]`,
  **1.25 %**, cos **0.999922** (every block's worst is its `ffn.net.2`, i.e. the `K = 8960` reduction).
- **Saturation is not the limit**: at most **0.070 %** of a GEMM's weights clip at ±127, because each
  channel's scale comes from that channel's own maximum by construction.
- **Per-channel scale spread within one GEMM runs 1.8× to 79.8×** — that spread is the mechanism
  `RESULTS_scales.md` measured: one per-tensor scale pays **2.15×** this error (3.01 % vs 1.40 % on
  GEMM outputs) and fails the 0.999 cosine bar on 5/40 instances, which per-column never does.
- **FP16 remainder costs 0.028 %** rel. Frobenius worst-case over **525 tensors** (bias / norm /
  AdaLN / io weights — FP16 by the locked design, 0.02 % of parameters). 51.52 MiB payload.
- Two different quantities, do not quote interchangeably: `RESULTS_scales.md` measured GEMM
  **output** error with INT8 activations (per-column 1.40 % mean); the 0.93 % here is **weight-only**
  reconstruction.

## A filter bug this pass caught (worth keeping in the paper's deviations list)

The first version picked the block's FP16 remainder by **name suffix**, and `scale_shift_table`
matched only through a `len(shape) > 2` guard — so all **30 per-block AdaLN tables** ([1,6,1536],
9,216 params each = **276,480 params, 540 KiB**) were written to neither the INT8 region nor the
FP16 file. The script still printed PASS: the verifier only checked what the manifest listed, and
the manifest listed only what the writer wrote. That is the whole failure mode — a verifier that
trusts the manifest's tensor list cannot notice what the manifest forgot.

Fix: classification is now **by role** (`inspect_weights.role_of`) in both the block and top-level
loops, and both scripts assert coverage — **1317 checkpoint tensors = 300 INT8 + 525 FP16 + 492
expert-side deferred, 0 in no file**. The 525 now verified include the 5-D `patch_embedding.weight`,
every `condition_embedder.*.weight`, `proj_out.weight` and both `scale_shift_table`s.

## A figure the manifest contradicts (reported, not applied)

`configs/block_manifest.json` records `int8_bytes_per_block = 49,586,432` (47.29 MiB), which counts
the block's **rank-128 LoRA factors as INT8**. The locked design stores those factors **FP16 and
separate, never merged into the INT8 base** — so the INT8 region is **44.25 MiB**, and the factors
are a further **240,089,216 B (229.0 MiB)** of FP16 for the whole model (120,044,608 params).
47.29 MiB stays the right *ceiling* for a region that would have to hold both; the artifact as built
is 44.25 MiB INT8 + 98 KiB FP16 per block. Nothing here reopens a decision.

## Leg 4 — end-to-end PSNR/SSIM: **PASS** (run 2026-11-16 05:10:22)

The environment question is closed (`.venv-ref`, torch 2.15.0+cpu / diffusers 0.43.0, commit
`8663c3f`), and the leg has now run. Tool `phase1_quant/psnr_gate1.py`, run record
`docs/device/RESULTS_gate1_psnr.md`, capture `gate1_psnr_20261116_051022.json`.

**Method.** Load the reference `AutoencoderKLWan` in fp32; decode a fixed seeded latent
`[1, 16, 3, 34, 60]` (Preset A → 9 × 3 × 272 × 480); then rebuild **every decoder weight** from
`artifacts/vae/` (the INT8 bytes × the artifact's own per-output-channel scales, OHWI per tap,
inverted to `[Cout, Cin, kT, kH, kW]`) and decode the same latent through the same code path. One
pass is **30.5 s** for 5,117.3 GMAC on this CPU — the decode is cheap enough to repeat.

| config | layers quantised | PSNR | SSIM | verdict |
|---|---|---|---|---|
| **all 36 quantised (the locked scheme, whole decoder)** | 36/36 | **40.22 dB** | **0.9708** | **PASS** |
| quantised except the two worst weight-cosine layers | 34/36 | 48.99 dB | 0.9962 | PASS |
| Gate 1 bar | — | ≥ 28 dB | ≥ 0.92 | — |

**The inversion is asserted, not assumed.** Each rebuilt weight must reproduce the artifact
manifest's recorded per-layer `rel_fro`; all **36/36** do, to a max |diff| of **4.5e-07**. A
transposed or tap-shifted rebuild still looks quantised, so this is the check that the number is
the artifact's and not an accident.

**Attribution — the loss concentrates in two layers, not the scheme.** Leaving
`up_blocks.{0,1}.upsamplers.0.time_conv` in fp32 (weight cosine **0.9974**, the two layers P3.8a
already flagged as the locked per-column scheme's casualties) moves 40.22 → **48.99 dB** and
0.9708 → 0.9962: those **two of thirty-six layers carry 8.77 dB** of the end-to-end loss. So the
cost is *localised* — a per-output-column scale on a 3×1×1 temporal conv — rather than a diffuse
cost of INT8, which is the ground on which the per-`(channel, tap)` alternative (worst layer
0.999910) should be judged. Not claimed: that 48.99 dB is the fp32 decoder. The residual 8.77 dB
above it is the other 34 layers' contribution and is not decomposed here; one more config per layer
group would price it, and that is the student's call to spend.

Per-frame (all-quantised): f0 53.49, f1 38.47, f2 43.21, f3 41.85, f4 40.89, **f5 37.88 (min)**,
f6 39.33, f7 39.45, f8 39.13 dB. Frame 0 is the causal boundary frame — cheap to predict, hence the
outlier in the good direction.

**Scope that must survive into the paper.** This is **weight-only**: activations stay fp32 because
the activation scale `SA` is `calibrate.py`'s output and nothing may invent it. The device's INT8
activation path is a *different* and additional cost; this number answers "does the locked weight
quantiser preserve the decoder", which is what Gate 1's bar asks.

**Harness bug caught en route, and why it mattered.** The first version cloned the *quantised*
tensors as its "pristine fp32" copy, so the fp32-restore step re-applied INT8 and the two configs
produced **bit-identical** output (PSNR 40.2159098895 dB both, `SSIM` 0.9708006514 both). The
headline number was still valid (the fp32 reference is decoded before any patching), but the
attribution was fiction. Now every config's output is hashed and a repeat hash **raises** instead
of printing a plausible table. Same class of failure as the P3.8a coverage bug: a plausible table
with nothing behind it.

Next: `calibrate.py` (the only source of `SA`, which is what turns "executes" into "decodes"), then
`phase1_quant/expert_factors.py` / `vae_export.py` are already done — see `RESULTS_phase1_artifacts.md`.
