# Open questions — decisions the student owns

## Answered by the student (2026-11-15) — locked

| Q | decision |
|---|---|
| Q1 workload | **Drop the 1-start-frame + 8-synthesised VFI logic entirely.** Workload is 9-frame **text-to-video** stream synthesis: `F_z = 3` latent temporal positions, conditioned only on **umT5 text embeddings precomputed offline on the host** (the text encoder never consumes Pixel 6a RAM). |
| Q2 device | Pixel 6a to be attached; Gate 0 runs as soon as `adb devices` lists it. |
| Q3 resolution | **Preset A (480×272, 9 frames, `3×34×60`, `M = 1,530`, ~28 MiB attention activation per block)** is the primary target. Use 256×256 only as a fast loop; 832×480 is opt-in. |
| Q4 second engine | **Always commit to the measured closed-form cut `k_b* ≥ 1`**, even when it is small (`[5,25]`) and the speedup is modest (1.05–1.15×); explain the shift with Eq. 4 as a systems finding. If NNAPI refuses Edge-TPU delegation (all ops on `cpu`, `T_N ≫ T_G`), **invert the roles** (faster engine takes the prefix, slower takes suffix+VAE) with the same period equation — do not substitute a third engine unannounced. |
| Q5 publishing | **Public repo from the start**: https://github.com/harris11ax/HeteroVDiff_Pixel_6a |

---

## Earlier open questions (kept for the record)

Each item had a recommended default; the repo was configured to assume it while answers were pending.

---

## Q1. The workload as specified does not exist — which reframing?

Blueprint §1 rows 3 & 6 describe **frame interpolation**: "9-frame chunks, 1 conditioning start
frame + 8 synthesized". The DCM-Wan2.1 checkpoint is **text-to-video only** — `transformer/config.json`
has `image_dim: null`, `added_kv_proj_dim: null`, and `modeling_wan.py` builds no image embedder
(verified). There is no code path that consumes a conditioning frame.

- **A (recommended).** Reframe as a **9-frame T2V chunk stream**. Keeps `B=180`, the cut, every
  measurement; drops the word "interpolation". Honest and fully supported by the weights.
- **B.** Keep the interpolation *presentation*: generate 9-frame chunks with stride 4 and cross-fade
  the seam, so the output reads as interpolation between real clips. Costs a wrapper script, not a
  model change. Still T2V underneath — must be stated as such.
- **C.** Switch to an image-conditioned Wan (I2V-14B). 14 B params ⇒ ~3.7 GiB INT8 base, does not
  fit a 6 GB phone with activations. Not viable, listed for completeness.

## Q2. Is the Pixel 6a available to attach now?

Everything from Gate 0 onward is blocked on it (`python tools/probe/probe_host.py --preflight`
currently reports `NONE ATTACHED`). The probe binary is built and waiting.

- **A (recommended).** Attach it on USB now, I run Gate 0 and record the capability profile.
- **B.** Not now — proceed with Phase 1 (quantiser + host-side parity), which needs no phone.

## Q3. Primary resolution target for the measurements?

- **A (recommended).** 256×256 baseline first for iteration, **Preset A (480×272)** as the reported
  result, attempt Preset B (832×480) and report it as the paper-matching config if it fits.
- **B.** Preset A only (skip the baseline) — faster to a result, less headroom when debugging.
- **C.** Preset B primary (832×480, `M=4680`) — closest to the paper, but a 1 GiB score matrix if
  attention is not de-fused, and long VAE decodes per chunk.

## Q4. If Gate 0 shows the Mali GPU is not a viable second stage, what do we commit to?

The paper's premise is NPU ≈ 3.4× faster per block than GPU. On Tensor G1 that ratio is unknown and
could invert. This decision only matters if the measurement is bad — but it changes the architecture.

- **A (recommended).** Decide from the measurement. If `T_G ≫ T_N` such that the GPU would own < ~10
  blocks, switch Process A to the **CPU cluster (XNNPACK INT8, 2×X1+2×A76)** and keep the design
  (two processes, one INT8 copy, one block-level cut, co-run rates). Report the substitution.
- **B.** Commit to Mali + Edge TPU regardless; if it loses, report the negative result (the course
  accepts this, and it is the literal blueprint).
- **C.** Run **both** second-engine candidates and report a three-way comparison (more work, stronger
  paper).

## Q5. Publishing

- **A (recommended).** Local git only for now; push once Phase 1 gate passes (avoids committing a
  half-built tree to a repo the course may grade).
- **B.** Create `github.com/harris11ax/HeteroVDiff_Pixel_6a` now (private). Requires installing `gh`
  or a PAT — neither is on this machine today.
- **C.** Public from the start.

---

## Answered by me (no input needed, change only if you disagree)

| item | decision | why |
|---|---|---|
| Quantisation scope | INT8 for linear weights only; FP16 for bias/norm/AdaLN/embeddings | 0.02 % of params, and AdaLN gates are extremely sensitive |
| Weight layout | row-major `[N_out,K]`, no HMX 32×32 tiling | no HMX on Pixel; row-major is native for NNAPI `FULLY_CONNECTED` *and* gives the `char4` operands Mali wants |
| NPU executor form | NNAPI graph builder with the paper's `run_blocks()` interface | Pixel exposes no custom-kernel path; interface keeps coordinator identical |
| Zero-copy default | try AHB+`SCM_RIGHTS`, fall back to file-backed `MAP_SHARED` | route B preserves "one stored copy" without vendor-specific IPC |
| Attention | de-fused, 256-row score block | 1 GiB score matrix at Preset B otherwise |
| VBench-2.0 | out of scope for now | course needs PSNR/SSIM/LPIPS; VBench-2.0 needs detector/VLM checkpoints and days of runtime |
| Repo layout | phase-numbered dirs, `configs/` lockfiles, `results/raw` gitignored | every reported number traceable to a JSON |
