# Phase 1 deliverables 2 and 3 — the expert factor artifact and the VAE execution plans

*Host only, 2026-11-15 20:5x. No device number appears in this document; the two rates quoted are
re-bound from the gate-compliant `T_V` sweeps and the script refuses to finish unless its own walk
reproduces them exactly. Reproduce with*

```
python phase1_quant/expert_factors.py && python phase1_quant/verify_experts.py
python phase1_quant/vae_export.py
```

Deliverable 1 (INT8 base + Gate 1) is `RESULTS_gate1.md`. This covers **deliverable 2**
(`expert_factors.py` — the detail expert, FP16, factorised) and **deliverable 3** (`vae_export.py` —
`AutoencoderKLWan` flattened into an executable plan for both engines). Deliverable 4
(`calibrate.py`) is still blocked, see the end.

## Deliverable 2 — the detail expert's factors

`artifacts/experts/detail/` (gitignored, regenerable in 4.5 s):

| Path | Content | Bytes |
|---|---|---|
| `block_{00..29}.fp16.bin` | one block's 8 attention projections × (`A [128,1536]`, `B [1536,128]`), FP16, 4096-B aligned | 6,291,456 each |
| `top.fp16.bin` | `condition_embedder_lora` (10 tensors) + `proj_out_lora` (2) | 51,345,536 |
| `manifest.json` | per-tensor offset, shape, rank, K/N_out, FP16 parity, sha256 |
| `routing.json` | steps → expert: `[semantic×3, detail×3]`, 180 block executions |

- **240,089,216 B FP16 (228.97 MiB), delta 0** against the Gate 1 figure in
  `design_locks.json:expert_factors_mib_fp16 = 229.0`. 492 tensors = 30 × 16 + 12, which is exactly
  every expert-side tensor in the checkpoint.
- **FP16 parity**: worst rel. Frobenius **0.0220 %**, min cosine **1.0000000** — the payload is the
  checkpoint rounded to FP16, nothing else.
- **One stored INT8 copy of the base weights is intact**: the artifact writes **0 INT8 bytes**, and
  the run asserts its tensor set is *disjoint* from the INT8 base's 300 GEMM names. The factors are
  not merged, per the locked design.
- **`ffn` has no LoRA in this checkpoint** — 480 block LoRA tensors / 30 blocks = 16 = 8 projections
  × (A,B), and the FFN weights are plain `.weight` (shared base). Checked by count, not assumed.
- **`lora.scaling_alpha` is `null` on purpose.** There is no alpha or scale tensor in the
  checkpoint, only `.weight`, so the `alpha/rank` runtime constant must be read off the reference repo.
  Recording a guess here would silently rescale every projection in the pipeline.
- The decision this artifact carries is already measured: `design_locks.json`
  `mali_int8_gemm.lora_delta_ms_per_block` — **+226 ms/block** if the factors are computed FP16,
  **+41 ms** if they were INT8, **0 ms** if pre-merged into the epilogue. Building the factors does
  not choose; it makes all three runnable.

Independent check (`verify_experts.py`, which re-implements the role classifier rather than importing
it, so a classifier bug cannot verify itself):

```
files verified    : 31 (240,099,328 B on disk, 240,089,216 B payload)
tensors verified  : 492 (492 expert-side in checkpoint, 0 unwritten, 0 foreign)
byte identity     : PASS — payload re-derived from FP32 source
VERDICT: PASS
```

The 10,112 B of file-vs-payload difference is 4096-B alignment padding (the 1-D biases), the same
distinction `RESULTS_gate1.md` draws for the FP16 remainder.

## Deliverable 3 — the VAE decoder as two plans

`configs/vae/` (**committed**: small, deterministic, and the Phase 3 executors' input — the same
precedent as `configs/vae_decode_ops.json`).

| File | Content |
|---|---|
| `layers.json` | 37 ops in module order (36 convs + 1 full 3-D attention), each with weight tensor + shape, stride, `causal_pad_T = [kT-1, 0]`, **frames of input to cache across a chunk boundary**, in/out `(f,h,w)`, MACs, and the measured class it binds to |
| `plan_npu.json` | every `kT=3` causal conv → `kT` shifted 2-D `CONV_2D` taps, each with its weight byte-slice and the input frame it reads; **N = 1**, int32 accumulate then dequant; 592 executes/chunk |
| `plan_mali.json` | the whole 3-D conv as one `conv8_taps` dispatch; `K`-total = `nst × CinP`, work-group count, the weight-row repack spec (`k = ((t·s+ky)·s+kx)·CinP + c`), 216 dispatches/chunk |
| `scaling.json` | `latents_mean` / `latents_std` (16 ch) and `z_phys = z·std + mean` |
| `tiles.json` | tile schedule, halo (1 px spatial, 2 frames temporal), per-level grids |

**The export is checked against the measurement, not trusted.** Three assertions run on every
invocation:

1. its layer set and per-layer MACs equal `configs/vae_decode_ops.json` exactly — the inventory the
   `T_V` sweep was taken against: **37 ops, 5149.68 GMAC, MATCH**;
2. walking the fresh header reproduces the output grid **9 × 272 × 480** from the latent
   `3 × 34 × 60`, i.e. the causal/upsample rules are validated, not assumed;
3. binding the measured per-class rates back in reproduces **10.554 s/chunk on the TPU** and
   **12.256 s/chunk on Mali** — the two committed `T_V` figures, to the millisecond.

```
layers              : 37 (36 convs + 1 3-D attention), output [9, 272, 480] (OK)
MACs                : 5149.68 GMAC vs inventory 5149.68 GMAC (MATCH)
measured, bound in  : TPU 10.554 s/chunk (ref 10.554), Mali 12.256 s/chunk (ref 12.256)
executes            : TPU 592 CONV_2D executes/chunk (N=1), Mali 216 dispatches/chunk
```

Two things this pass states rather than smooths over:

- **`decoder.mid_block.attentions.0` has no measured rate on either engine** (6120 tokens, 384 ch,
  32.4 GMAC). It is the only layer in `layers.json` with `measured = null`, and it is why `T_V` stays
  a lower bound in both directions. Its share is 0.63 % of decode MACs.
- **The "60×40-latent-pixel tile schedule" in `docs/05_phase_plans.md` is unsatisfiable on Preset A
  and degenerates to one tile.** The latent plane is 34 × 60 = 2040 pixels, smaller than the 2400-pixel
  tile, so there are no seams to manage at this preset — recorded in `tiles.json:spec_tile_note`
  instead of inventing a grid the geometry does not have. At Preset B (60 × 104 latents) the tiling
  question is real again and needs a re-export with that preset's table.

## Still open in Phase 1

| deliverable | state |
|---|---|
| 1. INT8 base + Gate 1 | **done** (`RESULTS_gate1.md`); the PSNR/SSIM leg is blocked on torch/diffusers |
| 2. expert factors | **done and independently verified** (240,089,216 B FP16, delta 0) |
| 3. VAE export | **done** — both plans + scaling + tiles, cross-checked against the measured inventory and both `T_V` totals |
| 4. `calibrate.py` | **blocked**: it must *run* the FP32 reference forward pass to record `max|x|/127` per (step, block, GEMM), and this host has no `torch`/`diffusers`. Writing the script without the capability it needs would produce an artifact nobody can distinguish from a fabricated one, so the file is not written yet |

Next for Phase 1: install torch (CPU wheel) or land the FP32 reference path from Phase 4.1, then
`calibrate.py` writes `artifacts/act_scales.json` and the PSNR/SSIM leg unblocks with the same
install.
