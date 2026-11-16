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

## Still open (Gate 1 leg 4)

The end-to-end FP32-vs-INT8-simulated forward on one 9-frame 256×256 prompt (PSNR ≥ 28 dB,
SSIM ≥ 0.92) **has not run**. Blocker: this host has no `torch` and no `diffusers`
(`import torch` → `ModuleNotFoundError`), and the reference path needs the Wan transformer + umT5 +
`AutoencoderKLWan` to execute at all. Options in cost order: (a) `pip install torch --index-url
<cpu wheel index> diffusers` (~2.5 GB; 9 frames at 256×256 on CPU is minutes); (b) reuse the
authors' pipeline if it runs CPU-only; (c) numpy FP32 forward — large, only if torch is unavailable.

Next: `phase1_quant/expert_factors.py` (deliverable 2, expects 240,089,216 B FP16) and
`vae_export.py` (deliverable 3), then the PSNR leg.
