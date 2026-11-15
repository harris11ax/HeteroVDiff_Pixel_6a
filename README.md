# HeteroVDiff on a Google Pixel 6a

A reproduction of [HeteroVDiff](https://arxiv.org/abs/2610.16244) — pipelining a multi-expert video
diffusion transformer across a phone's NPU and GPU from **one stored INT8 copy of the base
weights** — retargeted from the paper's Galaxy S25 (Snapdragon 8 Elite) to a **Pixel 6a
(Google Tensor G1)**.

ECE 4501/6501 (Co-design for ML) final project, University of Virginia.

## What is being built

| phase | artifact |
|---|---|
| 1 | per-output-column INT8 quantiser, per-DiT-block weight regions, factorised detail-expert factors, VAE layer export |
| 2 | on-device capability probe + co-run rate measurement (`T_G`, `T_N`, `T_V`) + closed-form cut solver |
| 3 | Mali-G78 OpenCL INT8 executor, NNAPI (Edge TPU) suffix executor, two-process zero-copy coordinator |
| 4 | throughput / SLO / memory / PSNR-SSIM-LPIPS evaluation against an FP32 reference |

The cut that this project reproduces (paper §4.3):

```
B = N·L = 180 block executions per chunk
period(k_b) = max(k_b·T_G, (B − k_b)·T_N + T_V)
k*          = (B·T_N + T_V) / (T_N + T_G)
```

## Start here

- `HANDOFF.md` — current state, blockers, ordered next actions
- `docs/00_START_HERE.md` — the reading order
- `docs/03_spec_vs_reality.md` — where the original blueprint does not survive contact with Tensor G1

## Requirements

Android SDK (platform-tools, NDK 27.3.13750724, cmake 3.22.1), Python 3.11 with
`huggingface_hub`/`numpy`, and a Pixel 6a with USB debugging. Model weights are **not** in the repo:

```python
from huggingface_hub import snapshot_download
snapshot_download("cszy98/DCM", local_dir="weights_raw",
                  allow_patterns=["DCM_WAN/transformer/*", "DCM_WAN/vae/*",
                                  "DCM_WAN/text_encoder/*", "DCM_WAN/tokenizer/*",
                                  "DCM_WAN/scheduler/*", "DCM_WAN/model_index.json"])
```

## Quick start

```bash
python phase1_quant/inspect_weights.py --manifest configs/block_manifest.json   # checkpoint facts
python tools/probe/probe_host.py --preflight                                   # is the phone attached?
python tools/probe/probe_host.py --build                                       # Gate-0 device probe
python tools/probe/probe_host.py --cooldown 30                                 # thermal gate
```

## Reference

Zheyu Shen, Ziyao Wang, Yexiao He, Guoheng Sun, Yuning Zhang, Yiting Wang, Wanghao Ye, Ang Li.
*HeteroVDiff: Pipelining Video Diffusion across Mobile NPU and GPU with Shared Weights.*
arXiv:2610.16244v1 (2026).

Model: Lv et al., *Dual-Expert Consistency Model for Efficient and High-Quality Video Generation*
(DCM), ICCV 2025 — code `Vchitect/DCM`, weights `cszy98/DCM`.
