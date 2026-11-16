#!/usr/bin/env python3
"""P3.13 — one `CONV_2D` layer, both ends dumped: is the Edge TPU doing the conv?

The wired decode executes (P3.12) but does not reproduce the fp32 decode (P3.13's
end-to-end PSNR). Two explanations remain and they have opposite fixes:

  * the HAL's `CONV_2D` output is not the convolution  -> a Gate-3-class device hole
  * the program between the convs is wrong             -> a host wiring bug

This separates them on a single layer with **no wiring at all**: `npu_block convwire
<...> 1 <L>` writes that conv's exact int8 input codes and output codes, so the host can
emulate the taps from the same bytes.

Three quantities, in the order that makes a miss unambiguous:

  1. `emulated vs checkpoint` — the artifact's INT8 taps dequantised and re-convolved by
     hand must reproduce `vae.decoder.<layer>` from the checkpoint weights. This is the
     guard on the host side: if it fails, the emulation is wrong and nothing else here
     means anything.
  2. `device vs emulated(correct)` — the conv as the model defines it (one int32
     accumulation over all temporal taps, bias once, one requant).
  3. `device vs emulated(device semantics)` — the conv as THIS program computes it: one
     graph per temporal tap, each graph rounding to int8 at the layer's SA_out and
     carrying the same int32 bias, then the taps summed as int8. Reproducing 3 exactly
     means the HAL is faithful and the remaining error is the program's.

    .venv-ref/Scripts/python.exe tools/npu/conv_parity.py --layer 0 --dump artifacts/vae/dump_layer_0_<ts>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "phase1_quant"))
import psnr_gate1 as G1  # noqa: E402  load_artifact / rebuild_layer

ART = REPO / "artifacts" / "vae"
VAE_DIR = REPO / "weights_raw" / "DCM_WAN" / "vae"
OUT = REPO / "docs" / "device"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_meta(p: Path) -> dict:
    d = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition("=")
        if not k:
            continue
        try:
            d[k] = float(v)
        except ValueError:
            d[k] = v
    for k in ("f", "c", "h", "w", "bytes"):
        if k in d:
            d[k] = int(d[k])
    return d


def codes_fhwc(p: Path, meta: dict) -> np.ndarray:
    """int8 [f][h][w][c] on disk -> float [f, c, h, w] at the declared scale."""
    raw = np.fromfile(p, dtype=np.int8)
    want = meta["f"] * meta["c"] * meta["h"] * meta["w"]
    if raw.size != want:
        raise SystemExit(f"{p.name}: {raw.size} codes vs {want}")
    a = raw.reshape(meta["f"], meta["h"], meta["w"], meta["c"]).transpose(0, 3, 1, 2)
    return a.astype(np.float32) * float(meta["sa"])


def stats(ref: torch.Tensor, got: torch.Tensor) -> dict:
    d = got - ref
    rmse = float(torch.sqrt(torch.mean(d ** 2)))
    rms = float(torch.sqrt(torch.mean(ref ** 2)))
    return {"rel_rmse": rmse / max(rms, 1e-12), "max_abs": float(d.abs().max()),
            "cosine": float(torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(),
                                                                  dim=0)),
            "ref_rms": rms, "got_rms": float(torch.sqrt(torch.mean(got ** 2)))}


def code_cmp(ref_codes: np.ndarray, got_codes: np.ndarray) -> dict:
    """Both int8 arrays, same layout: how many codes agree, and by how much they miss."""
    ref_codes = np.asarray(ref_codes).ravel()
    got_codes = np.asarray(got_codes).ravel()
    if ref_codes.size != got_codes.size:
        raise SystemExit(f"code_cmp: {ref_codes.size} vs {got_codes.size} codes")
    d = got_codes.astype(np.int16) - ref_codes.astype(np.int16)
    n = d.size
    return {"codes": int(n), "bit_exact_pct": 100.0 * float((d == 0).sum()) / n,
            "within1_pct": 100.0 * float((np.abs(d) <= 1).sum()) / n,
            "mean_abs_code_err": float(np.abs(d).mean()),
            "max_abs_code_err": int(np.abs(d).max()),
            "pct_ref_saturated": 100.0 * float((np.abs(ref_codes.astype(np.int16)) >= 127).sum()) / n,
            "pct_got_saturated": 100.0 * float((np.abs(got_codes.astype(np.int16)) >= 127).sum()) / n}


def quant(x: torch.Tensor, sa: float) -> np.ndarray:
    """torch float [C, F, H, W] -> int8 [f,h,w,c] at `sa` (the harness's own rounding)."""
    q = torch.clamp(torch.round(x / sa), -127.0, 127.0).to(torch.int16).numpy()
    return np.ascontiguousarray(q.transpose(1, 2, 3, 0)).astype(np.int8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--dump", required=True, help="directory with vae_dump_{in,out}_lNN.*")
    ap.add_argument("--out", default="conv_parity")
    args = ap.parse_args()
    d = Path(args.dump)
    tag = f"l{args.layer:02d}"
    imeta = read_meta(d / f"vae_dump_in_{tag}.txt")
    ometa = read_meta(d / f"vae_dump_out_{tag}.txt")
    x = codes_fhwc(d / f"vae_dump_in_{tag}.bin", imeta)      # [f, cin, h, w]
    dev_out = codes_fhwc(d / f"vae_dump_out_{tag}.bin", ometa)
    dev_codes = np.fromfile(d / f"vae_dump_out_{tag}.bin", dtype=np.int8)
    log(f"layer {args.layer}: in {imeta['f']}x{imeta['c']}x{imeta['h']}x{imeta['w']} "
        f"sa_in={float(imeta['sa']):.6g} -> out {ometa['f']}x{ometa['c']}x{ometa['h']}x"
        f"{ometa['w']} sa_out={float(ometa['sa']):.6g}")

    man, wbytes, scales = G1.load_artifact()
    lay = man["layers"][args.layer]
    W = G1.rebuild_layer(lay, wbytes, scales)                # [Cout, Cin, kT, kH, kW] fp32
    sc = scales[lay["scales"]["offset"] // 4:
                lay["scales"]["offset"] // 4 + lay["scales"]["count"]]
    bi = np.fromfile(ART / "bias_f32.bin", dtype="<f4")
    bias = bi[lay["bias"]["offset"] // 4: lay["bias"]["offset"] // 4 + lay["bias"]["count"]] \
        if "bias" in lay else np.zeros(lay["scales"]["count"], dtype=np.float32)
    log(f"artifact: weight {W.shape}, {len(lay['taps'])} taps, bias {bias.shape} "
        f"(scales {sc.min():.3g}..{sc.max():.3g})")

    taps = sorted(lay["taps"], key=lambda t: int(t["tap"]))
    Wt = torch.from_numpy(W)                                 # [Cout, Cin, kT, 3, 3]
    xt = torch.from_numpy(x).permute(1, 0, 2, 3).unsqueeze(0)  # [1, Cin, F, H, W]
    bt = torch.from_numpy(bias.astype(np.float32))
    F = x.shape[0]
    sa_in, sa_out = float(imeta["sa"]), float(ometa["sa"])

    def conv_taps(out_scale: float | None, per_tap_round: bool,
                  order: str = "fwd") -> torch.Tensor:
        """Temporal-tap conv, causal (zero for src < 0). per_tap_round reproduces the
        device: each tap's graph rounds to int8 at sa_out and adds the same bias, and
        the tap outputs are summed as int8 with a final clamp. out_scale=None returns the
        unquantised fp32 accumulation (the host guard against the checkpoint).

        `order` is the pairing between the artifact's tap index and the frame it reads.
        PyTorch's causal conv3d pads 2 zeros at the FRONT, so weight slice 0 is applied to
        x[f-2]: "rev" is that pairing (tap t <-> frame f-(kt-1-t)). "fwd" is the plan's
        own label (tap t <-> f-t). Which one is right is measured below, not assumed.
        """
        nT = len(taps)

        def tap_delta(t: dict, i: int) -> int:
            """fwd: the artifact's own `input_frame` label; rev: mirrored."""
            s = str(t.get("input_frame", "f"))
            d = 0 if s in ("f", "") else int(s.split("-", 1)[1].strip())
            return d if order == "fwd" else (nT - 1 - d)

        def tap_out(t: dict, i: int, with_bias: bool) -> torch.Tensor:
            """One temporal tap over ALL frames: [1,Cin,F,H,W] -> [1,Cout,F,H,W], causal."""
            delta = tap_delta(t, i)
            src = torch.zeros_like(xt)
            lo = max(0, delta)
            if lo < F:
                src[:, :, lo:] = xt[:, :, lo - delta: F - delta]
            o = torch.nn.functional.conv2d(src[0].permute(1, 0, 2, 3),     # [F, Cin, H, W]
                                           Wt[:, :, int(t["tap"])],
                                           bias=(bt if with_bias else None), padding=1)
            return o.permute(1, 0, 2, 3).unsqueeze(0)                       # [1, Cout, F, H, W]

        if out_scale is None:
            acc = torch.zeros((1, Wt.shape[0], F, x.shape[2], x.shape[3]))
            for i, t in enumerate(taps):
                acc = acc + tap_out(t, i, with_bias=(i == 0))    # bias once: the model's conv
            return acc
        if per_tap_round:
            # the device's own semantics: every tap graph carries the same INT32 bias
            outs = [torch.clamp(torch.round(tap_out(t, i, with_bias=True) / out_scale),
                                -127, 127) * out_scale for i, t in enumerate(taps)]
            acc = torch.zeros((1, Wt.shape[0], F, x.shape[2], x.shape[3]))
            for f in range(F):
                s = sum(o[:, :, f] for o in outs)
                acc[:, :, f] = torch.clamp(torch.round(s / out_scale), -127, 127) * out_scale
            return acc
        acc = torch.zeros((1, Wt.shape[0], F, x.shape[2], x.shape[3]))
        for i, t in enumerate(taps):
            acc = acc + tap_out(t, i, with_bias=(i == 0))
        return torch.clamp(torch.round(acc / out_scale), -127, 127) * out_scale

    # ---- 1. the host-side guard: emulation of the artifact vs the checkpoint module
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(str(VAE_DIR), torch_dtype=torch.float32).eval()
    name = lay["name"]
    mod = vae.get_submodule(name)
    with torch.no_grad():
        ref_ckpt = mod(xt).clone()
    guards = {}
    for order in ("fwd", "rev"):
        g = stats(ref_ckpt, conv_taps(out_scale=None, per_tap_round=False, order=order))
        guards[order] = g
        log(f"1. emulated taps ({order}) vs checkpoint `{name}`: rel-RMSE "
            f"{g['rel_rmse'] * 100:.3f} % | cosine {g['cosine']:.8f} | max|Δ| {g['max_abs']:.4g}")
    order = max(guards, key=lambda k: guards[k]["cosine"])
    g = guards[order]
    log(f"   -> tap order **{order}** reproduces the checkpoint; using it below")

    # ---- 2 and 3: compare the two emulations to the device's own codes
    emu_correct = conv_taps(out_scale=sa_out, per_tap_round=False, order=order)
    emu_devsem = conv_taps(out_scale=sa_out, per_tap_round=True, order=order)
    rows = []
    dev_t = torch.from_numpy(dev_out).permute(1, 0, 2, 3).unsqueeze(0)   # [1, C, F, H, W]
    for label, emu in (("device semantics (per-tap int8, bias per tap)", emu_devsem),
                       ("correct semantics (one accumulation, bias once)", emu_correct)):
        c = quant(emu[0], sa_out)
        cc = code_cmp(c, dev_codes)
        s = stats(dev_t, emu)
        rows.append({"emulation": label, **cc, **s})
        log(f"device vs {label}: {cc['bit_exact_pct']:.2f} % codes bit-exact | "
            f"within ±1 {cc['within1_pct']:.2f} % | mean|Δcode| {cc['mean_abs_code_err']:.2f} | "
            f"cosine {s['cosine']:.6f} | rel-RMSE {s['rel_rmse'] * 100:.2f} %")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    doc = {"kind": "conv2d_layer_parity", "timestamp": ts, "layer": args.layer,
           "layer_name": name, "dump": str(d),
           "io": {"in": imeta, "out": ometa, "taps": len(taps)},
           "host_guard_vs_checkpoint": g,
           "device_vs_emulations": rows,
           "note": "device codes are the dumped int8 the harness stored; emulations use the "
                   "artifact's own INT8 taps dequantised by its per-output-channel scales."}
    (OUT / f"{args.out}_{ts}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")

    md = [f"# `CONV_2D` single-layer parity — layer {args.layer} (`{name}`) — {ts}", "",
          f"in {imeta['f']}×{imeta['c']}×{imeta['h']}×{imeta['w']} at sa_in "
          f"{float(imeta['sa']):.6g} → out {ometa['f']}×{ometa['c']}×{ometa['h']}×"
          f"{ometa['w']} at sa_out {float(ometa['sa']):.6g}, {len(taps)} taps", "",
          "## 1. Host guard — the emulation vs the checkpoint", "",
          f"rel-RMSE **{g['rel_rmse'] * 100:.3f} %**, cosine **{g['cosine']:.8f}**, "
          f"max|Δ| {g['max_abs']:.4g} (this is the artifact's quantisation, nothing else)", "",
          "## 2. Device vs the emulations", "",
          "| emulation | bit-exact codes | within ±1 | mean abs Δcode | cosine | rel-RMSE |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['emulation']} | {r['bit_exact_pct']:.2f} % | {r['within1_pct']:.2f} % | "
                  f"{r['mean_abs_code_err']:.2f} | {r['cosine']:.6f} | "
                  f"{r['rel_rmse'] * 100:.2f} % |")
    md += ["", "Device saturated codes: "
           f"{rows[0]['pct_got_saturated']:.3f} %; emulation {rows[0]['pct_ref_saturated']:.3f} %.",
           ""]
    (OUT / f"{args.out}_{ts}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {OUT / f'{args.out}_{ts}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
