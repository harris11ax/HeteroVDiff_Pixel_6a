# START HERE — HeteroVDiff, reproduced on a Google Pixel 6a

You are the coding agent picking up this project. This repo is a **prepared** version of
`ECE 4501_6501 Final Project Specification & Agent Handoff Blueprint` (the blueprint),
targeting a reproduction of *HeteroVDiff: Pipelining Video Diffusion across Mobile NPU and
GPU with Shared Weights* (arXiv:2610.16244v1) on a **Pixel 6a (Tensor G1)**, not the paper's
Galaxy S25 (Snapdragon 8 Elite).

Everything below is either **verified on this machine / against the real checkpoint** or
explicitly marked **[UNVERIFIED]** with the probe that settles it.

## Read in this order

| # | File | Why |
|---|---|---|
| 1 | `docs/01_paper_digest.md` | What HeteroVDiff actually does, with the numbers you must match or explain |
| 2 | `docs/02_model_spec.md` | DCM-Wan2.1-1.3B as **measured from the checkpoint** (per-block, GEMM shapes, expert split) |
| 3 | `docs/03_spec_vs_reality.md` | ⚠️ Where the blueprint is wrong or will not port to Tensor G1, and the fallback ladder. **Read before writing code.** |
| 4 | `docs/05_phase_plans.md` | Phase-by-phase task list with acceptance gates |
| 5 | `docs/04_toolchain.md` | Exact paths/commands that work on this machine (NDK, cmake, adb, torch venv) |
| 6 | `docs/06_eval_protocol.md` | Metrics, configurations, SLO, thermal protocol, report tables |
| 7 | `OPEN_QUESTIONS.md` | Decisions the student still has to make — do not silently pick |

## State of preparation (what is already done for you)

| Item | Status |
|---|---|
| DCM-Wan2.1-1.3B transformer + VAE + umT5 text encoder + tokenizer | downloading to `weights_raw/DCM_WAN/` (HF `cszy98/DCM`, public) |
| Reference repos | cloned in `third_party/` — see `docs/07_repos.md` |
| Android NDK 27.3.13750724, CMake 3.22.1 + ninja, JDK 17 | installed (user-local, no admin) |
| On-device capability probe (OpenCL ext, AHB ceiling, NNAPI devices, zero-copy) | **built and compiling** → `tools/android/probe/build/probe`, driven by `tools/probe/probe_host.py` |
| Checkpoint inventory + per-block manifest | `phase1_quant/inspect_weights.py` → `configs/block_manifest.json` |
| Pixel 6a attached | ❌ **not yet** — every Gate 0 number is blocked on this |

## The one-paragraph summary of the design

Flatten the 6-step × 30-block DCM denoising trajectory into `B = 180` block executions.
Store the base weights **once** in INT8 (1.30 GiB measured, `1.30 GiB` base + 114.5 MiB expert
factors). Cut the flattened trajectory once at block index `k_b`, GPU runs `[0, k_b)`, NPU runs
`[k_b, 180)` plus the VAE decode; in a stream the GPU prefix of chunk `i+1` overlaps the NPU
suffix+decode of chunk `i`. Choose `k_b` from **co-run** rates `T_G, T_N, T_V`:

```
period(k_b) = max(k_b*T_G, (B - k_b)*T_N + T_V)        (paper Eq. 4)
k* = (B*T_N + T_V) / (T_N + T_G)                        (paper Eq. 5)
```

`N`, `L`, the quantisation, and the cut are algorithmically **exact** — every block runs in
order on full activations. The only inexactness is INT8 numerics, which the paper bounds at
−0.17 pp VBench quality and you must bound by PSNR/SSIM against an FP32 reference.

## Hard rules for this repo

1. **Measure before you model.** Every latency in the design comes from the phone, not from
   arithmetic intensity estimates. If the phone is not attached, write the harness and stop.
2. **One stored copy.** If a change creates a second copy of base weights on the device, it is
   wrong — it defeats the entire contribution (paper §4.2, §6.3).
3. **Never fabricate device numbers.** If `T_G/T_N/T_V` are unknown, the file says `null` and
   the table says `TBD (blocked: no device)`.
4. **The GPU is allowed to be the loser.** The paper's cut assumes the NPU is ~3.4× faster per
   block. On Tensor G1 that may not hold; then you report the measured ratio and the fallback
   in `docs/03_spec_vs_reality.md` — you do not tune the numbers to look like the paper.
5. All work lands in this repo, phase by phase, each phase leaving a runnable artifact.

## Repo map

```
phase1_quant/    INT8 per-column quantiser, block slicer, expert factor split, VAE graph export
phase2_profile/  ADB thermal protocol + co-run rate measurement + cut solver
phase3_pipeline/ Mali OpenCL INT8 kernels, NNAPI (Pixel NPU) suffix builder, C++ coordinator
phase4_eval/     FP32 reference + PSNR/SSIM/LPIPS + throughput/SLO/memory tables
tools/probe/     device capability probe (host driver)
tools/android/   NDK sources built as plain executables (no Gradle)
configs/         lockfiles: resolution presets, cut schedule, manifest
weights_raw/     downloaded checkpoints (gitignored)
third_party/     reference repos (gitignored, re-cloneable — see docs/07_repos.md)
```
