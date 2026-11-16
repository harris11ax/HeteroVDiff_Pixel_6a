#!/usr/bin/env python3
"""Proof that the 89x step between two consecutive convs is the norm, measured.

`convseq` chains layer i's int8 output into layer i+1's input by rescaling. The
calibration (P3.11) says that cannot be right for a resnet pair: measured
`mid_block.resnets.0.conv1` emits max|x| = 54.79 while `...resnets.0.conv2`
consumes max|x| = 0.612, a 89x gap no rescale produces.

`WanResidualBlock.forward` applies `norm2` = `WanRMS_norm` and then SiLU between
them. This script captures conv1's real output in one decode, applies that norm
by hand, and compares the result against the number the calibration measured on
conv2's input. If the two agree, the norm is the missing op and the numbers in
`vae_wiring.json` are the right thing to implement. If they do not, the wiring
plan is wrong and this prints the size of the disagreement instead of hiding it.

    python phase1_quant/check_norm_path.py [--resnet decoder.mid_block.resnets.0]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "weights_raw" / "DCM_WAN"
CAL = REPO / "artifacts" / "calib"
LATENT = (1, 16, 3, 34, 60)


def log(m: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet", default="decoder.mid_block.resnets.0")
    ap.add_argument("--calib", default="")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKLWan

    cal = Path(args.calib) if args.calib else sorted(CAL.glob("act_scales_vae_*.json"))[-1]
    C = json.loads(cal.read_text(encoding="utf-8"))
    rows = {r["layer"]: r for r in C["layers"]}
    conv2_in = rows[f"{args.resnet}.conv2"]["max_abs"]
    conv1_out_meas = rows[f"{args.resnet}.conv1"]["max_abs_out"]

    vae = AutoencoderKLWan.from_pretrained(str(MODEL / "vae"), torch_dtype=torch.float32).eval()
    blk = vae.get_submodule(args.resnet)
    conv1 = vae.get_submodule(f"{args.resnet}.conv1")

    cap: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        t = out if torch.is_tensor(out) else out[0]
        cap.setdefault("conv1_out", t.detach().clone())

    h = conv1.register_forward_hook(hook)
    g = torch.Generator().manual_seed(0)
    z = torch.randn(*LATENT, generator=g)
    with torch.no_grad():
        vae.decode(z)
    h.remove()

    x = cap["conv1_out"].float()
    log(f"{args.resnet}.conv1 output: shape {tuple(x.shape)}, max|x| = {x.abs().max():.4f} "
        f"(calibration recorded {conv1_out_meas:.4f})")

    # WanRMS_norm(images=False): F.normalize over dim=1, * sqrt(C), * gamma
    gamma = blk.norm2.gamma.detach().float()
    scale = blk.norm2.scale
    n = F.normalize(x, dim=1) * scale * gamma
    log(f"gamma: shape {tuple(gamma.shape)}, min {gamma.min():.4f} max {gamma.max():.4f}, "
        f"scale sqrt(C) = {scale:.4f}")
    log(f"after norm2      : max|y| = {n.abs().max():.4f}")
    s = F.silu(n)
    log(f"after silu       : max|y| = {s.abs().max():.4f}")

    measured = conv2_in
    got = float(s.abs().max())
    rel = abs(got - measured) / measured
    log(f"calibration measured on {args.resnet}.conv2 INPUT: {measured:.4f}")
    log(f"hand-applied norm+silu gives                  : {got:.4f}  (rel diff {rel * 100:.2f} %)")
    verdict = "PASS" if rel < 0.02 else "FAIL"
    log(f"{verdict}: the norm+SiLU is the step between the two convs"
        if rel < 0.02 else
        f"{verdict}: the norm does NOT reproduce the measured consumer range "
        f"({rel * 100:.1f} % apart) — the wiring plan is wrong, do not implement it")
    return 0 if rel < 0.02 else 1


if __name__ == "__main__":
    raise SystemExit(main())
