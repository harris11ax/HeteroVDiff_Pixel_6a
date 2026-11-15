# HANDOFF — HeteroVDiff on Pixel 6a

*Prepared 2026-11-15. This file is the orders sheet: read it first, then `docs/00_START_HERE.md`.*

## Mission

Reproduce HeteroVDiff's design on a **Pixel 6a (Tensor G1)**: one INT8 copy of DCM-Wan2.1-1.3B's
base weights shared by two engines, a single block-level cut of the `B = 180` denoising trajectory,
and a two-stage chunk pipeline. Course deliverable = working pipeline + measured tables + IEEE
report; the student owns the hardware rig, slides and paper.

## Where things stand

| | |
|---|---|
| Repo | `C:/Users/harri/Documents/GitHub/HeteroVDiff_Pixel_6a` (git initialised, first commit made) |
| Weights | `weights_raw/DCM_WAN/` — transformer 5.73 GiB FP32, VAE 508 MiB, umT5 text encoder 5 shards, tokenizer |
| Reference code | `third_party/` — DCM, Wan2.1, ncnn, llama.cpp(OpenCL), htp-ops-lib, ndk-samples, opencl-headers, VBench |
| Toolchain | NDK 27.3.13750724 + CMake 3.22.1 + adb installed user-local; JDK 17 at `C:/Users/harri/tools/jdk-17.0.21+9` |
| Device probe | **built** → `tools/android/probe/build/probe`, driver `tools/probe/probe_host.py` |
| Checkpoint manifest | `configs/block_manifest.json` (per-block tensor inventory, verified against the paper) |
| **Blocker** | **no Pixel 6a attached** → every Gate-0 measurement is pending |

## Ordered next actions

1. Attach the Pixel 6a, enable USB debugging, `python tools/probe/probe_host.py --preflight`.
2. `python tools/probe/probe_host.py --build` → commit `docs/device/probe_*.json` and `.md`.
3. Answer the recorded decisions in `configs/design_locks.json` (zero-copy route, NPU stage exists
   or not, second engine) using the probe output. **Do not guess these — each one is a fork in the
   rest of the project.**
4. Build Phase 1 (quantiser + manifest + Gate 1), which needs no phone: `docs/05_phase_plans.md`.
5. Only then Phase 2 measurements, Phase 3 code, Phase 4 tables.

## Decisions already locked (do not reopen without saying why)

- Backbone: **DCM-Wan2.1-1.3B**, `N = 6`, `L = 30`, `B = 180`; expert switch `κ = 3` (steps 0–2
  semantic, 3–5 detail) — verified in the checkpoint and in DCM's sampler.
- Quantisation: **per-output-column symmetric INT8**, one FP32 scale per column, base weights only;
  bias/norm/AdaLN/embeddings stay FP16.
- Detail expert stored **factorised** (rank-128 `A`,`B` + its own timestep block), never merged.
- One **block-level** cut, one handoff per chunk, `k_b` from co-run rates with `T_V`.
- Layout: **row-major `[N_out, K]` INT8** (HMX tiling dropped — no HMX on Pixel).
- The NPU stage is a **NNAPI graph**, not custom kernels (Pixel exposes nothing else).
- Attention is **de-fused** (QK → softmax → AV) with a 256-row score block.
- No fabricated device numbers, ever: unknown ⇒ `null` + "TBD (blocked)".

## Open decisions (asked, awaiting answers)

See `OPEN_QUESTIONS.md`. Until answered: assume the recommended option and mark it in
`configs/design_locks.json` with `"assumed": true`.

## Definition of done for the course project

1. Artifact + quantiser + parity report (Gate 1) committed.
2. Device probe + measured rates + solved cut (Gate 2) committed.
3. Two-process pipeline streaming ≥10 chunks on the phone without manual intervention (Gates 3–5).
4. `results/tables.md`: configurations, cut ablation, memory, quality (PSNR/SSIM/LPIPS), SLO sweep,
   plus a prediction-vs-measured table from Phase 2.
5. Report-ready list of deviations from the paper, each with the measurement that forced it
   (`docs/03_spec_vs_reality.md` is the draft).
