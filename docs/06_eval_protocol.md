# Evaluation protocol

Goal: produce the paper's **result shape** on Tensor G1, with every number traceable. The paper's
absolute numbers are not reproducible on different silicon — say so once, then report yours.

## Configurations (all consume the same INT8 artefact)

| id | DiT execution | decode | why it is in the table |
|---|---|---|---|
| `npu-only` | full 180 blocks on Process B | Process B | the strongest single engine ⇒ 1.0× reference |
| `gpu-only` | full 180 blocks on Process A (Mali) | Process B | shows the GPU's real rate; in the paper this is 0.35–0.48× |
| `dp` | whole chunks distributed to whichever engine is free | either | naive data parallelism — matches the paper's `DP`, and it *fails the SLO* there |
| `hetero-tensor` | per-GEMM column split, both engines each GEMM | Process B | the HeteroInfer-style baseline the paper beats |
| `hetero-cut-step` | one cut on a step boundary (`k_b ∈ {30,60,…}`) | Process B | ablation: step vs block granularity |
| `hetero-cut-block` | **the proposed design**, `k_b = k_b*` from co-run rates | Process B | the result |

`gpu-only` needs a decoder path on Process B even when Process A does the DiT — that asymmetry is
in the paper too (no Conv3d in the vendor GPU backend; here check whether Mali OpenCL is actually
fast enough for the VAE before assuming it).

## Workload

- 9-frame chunks (measured `F_z = 3`), prompts from the paper's suite or a fixed set of 10
  content-diverse prompts; prompt embeddings precomputed offline with the umT5 encoder, encoder not
  resident during generation (paper §6.1).
- Resolution: run **Preset A (480×272)** as the primary and **Baseline (256×256)** as the fast
  iteration target; attempt **Preset B (832×480)** and report it as the paper-matching config, with
  `M = 4680` (vs the paper's 20,280 — a deliberate difference because chunks are 9 frames not 49).
- Stream = 10 chunks back to back per configuration, discarding the first (pipeline fill).

## Metrics

**Throughput / latency**
- `period` = (time of chunk 10 − time of chunk 2) / 8 — steady state, after fill.
- `videos/hour = 3600 / period` (paper calls one chunk a "video"; keep the word `chunk` in your
  tables and note the mapping).
- `single-chunk latency` = launch → last decoded frame (report separately; the design optimises the
  period, not this).
- `SLO attainment` = fraction of chunks with latency ≤ `D`, where `D = 1.5 ×` the measured
  `npu-only` single-chunk latency at the same workload; also sweep `f ∈ {1.25,1.5,1.75,2,3}` as
  the paper does in its Figure 7.
- `speedup = period(npu-only) / period(config)`.

**Memory**
- tensor-bytes accounting (paper's method): base artefact + scratch + working buffers, per
  configuration. Report `GiB` to 2 dp.
- device PSS: `dumpsys meminfo` for each process at steady state + `dumpsys meminfo` total; also
  count distinct DMA-BUF inodes across the two processes (`/proc/<pid>/fdinfo/*`) to prove the
  single-copy claim. Report both; RSS is unreliable (page cache / compressed swap).

**Quality**
- PSNR, SSIM (skimage, `channel_axis=-1`, `data_range=1.0`) and LPIPS (AlexNet) between the
  INT8-arm frames and the FP32 reference frames of the same prompt/seed — per frame and mean.
- Gates from the blueprint: **SSIM ≥ 0.92, PSNR ≥ 28.0 dB** vs FP32. Record the value actually
  measured even if it fails; a failure with a diagnosis scores better than a silent retune.
- Also report drift *within* a stream (quality of chunk 1 vs chunk 10) — quantisation bugs often
  only show after the residual stream warms up.

**Cut ablation** (report as a table like the paper's Table 5): `k_b` at step boundaries, `k_b` from
solo rates, `k_b` from co-run rates without `T_V`, `k_b` from co-run rates with `T_V` ⇒ columns
GPU time / NPU time / idle / period / % over best.

## Thermal protocol (make it a script, not a habit)

1. `python tools/probe/probe_host.py --cooldown 30` — block until the hottest zone ≤ target.
2. Record ambient + all zone temps before and after (`docs/device/probe_*.json` does this).
3. Screen off (`adb shell input keyevent 26`), plugged into a powered hub, phone elevated.
4. Any run whose start temp exceeds the gate is discarded and re-run — log the discard.
5. Re-measure `T_V` on the day of the perf run (the paper saw it drift 186→270 s between days).

## Numbers you must be able to defend in the report

| claim | evidence file |
|---|---|
| "one stored copy of the base weights" | `results/raw/memory_*.json` + fdinfo inode count |
| "the cut is optimal among block boundaries" | `configs/cut_schedule.json` + `results/raw/cut_sweep.json` |
| "the pipeline beats NPU-only on stream throughput" | `results/raw/stream_*.json` (≥3 runs each) |
| "INT8 does not cost measurable quality" | `results/raw/quality_*.json` (PSNR/SSIM/LPIPS per frame) |
| "rates were measured co-running, not modelled" | `results/raw/rates_*.json` with the solo/co-run pairs |
| "deviation from the paper: X" | `docs/03_spec_vs_reality.md` + the probe JSON that forced it |
