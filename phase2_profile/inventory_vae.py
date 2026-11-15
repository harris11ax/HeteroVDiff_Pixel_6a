#!/usr/bin/env python3
"""Inventory the Wan `AutoencoderKLWan` **decoder** and compute the exact MAC
budget of one Preset-A chunk decode (T_V). No torch, header-only.

Why this exists
---------------
`T_V` (VAE decode) is the last rate blocking `k_b*`. Before any kernel is
written we need to know *where the MACs are*: the decoder is a causal 3-D conv
stack, and the paper's answer (QNN HTP `Conv3d`) does not exist on NNAPI. This
script produces the op inventory both candidate branches must serve:

  * every conv weight with its (Cin, Cout, kT, kH, kW)
  * causal temporal padding == (kT-1, 0) so `F_out == F_in`
  * per-layer output shape and MACs, walked in module order
  * the distinct *shape classes* (channel x spatial) -- the unit of measurement
    on the phone; one measured rate per class, summed like the T_N table

Shapes are walked, not guessed: `dim_mult` reversal, `F' = 2F-1` per temporal
upsample, `2x` nearest + 3x3 spatial resample, all read off the real tensors.

    python phase2_profile/inventory_vae.py
    python phase2_profile/inventory_vae.py --json configs/vae_decode_ops.json
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import OrderedDict, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_VAE = REPO / "weights_raw" / "DCM_WAN" / "vae" / "diffusion_pytorch_model.safetensors"

# Preset A (configs/design_locks.json -> workload.presets.preset_a)
PRESET_A = {"frames": 9, "H": 272, "W": 480, "F_z": 3, "H_z": 34, "W_z": 60}


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def conv_macs(shape, out_fhw, taps_used=None) -> int:
    """MACs of one conv: Cout x Cin x kT x kH x kW per output position.

    Accepts 4-D `[Co, Ci, kH, kW]` (the spatial resample convs are plain 2-D)
    as well as 5-D `[Co, Ci, kT, kH, kW]`.
    """
    if len(shape) == 4:
        co, ci, kh, kw = shape
        kt = 1
    else:
        co, ci, kt, kh, kw = shape
    taps = kt * kh * kw if taps_used is None else taps_used
    return co * ci * taps * out_fhw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=str(DEFAULT_VAE))
    ap.add_argument("--json", default=None)
    ap.add_argument("--preset", default="preset_a")
    args = ap.parse_args()

    if args.preset != "preset_a":
        raise SystemExit("only preset_a is wired (add the preset table here)")

    p = PRESET_A
    hdr = read_header(Path(args.vae))
    w = {k: v for k, v in hdr.items() if k != "__metadata__"}

    F, H, W = p["F_z"], p["H_z"], p["W_z"]
    ops = []          # ordered per-op records
    classes = OrderedDict()   # shape-class -> aggregated MACs / op count

    def record(kind, name, out_fhw, macs, cin=None, cout=None, k=None, nin_chunks=1,
               fhw=None):
        ops.append({"name": name, "kind": kind, "out_positions": out_fhw,
                    "macs": macs, "cin": cin, "cout": cout, "k": k,
                    "chunks": nin_chunks, "fhw": fhw})
        if kind == "conv":
            key = f"conv{tuple(k) if isinstance(k, list) else k}_c{cin}->{cout}@{out_fhw}"
            classes.setdefault(key, {"kind": kind, "cin": cin, "cout": cout,
                                     "k": list(k) if isinstance(k, list) else k,
                                     "out_positions": out_fhw,
                                     "macs": macs, "ops": 0,
                                     "fhw": list(fhw) if fhw else None})
            classes[key]["ops"] += 1

    def do_conv(prefix, f, h, ww, tap_decomp=1):
        """Causal 3-D conv (or a 2-D source conv) at spatial (f,h,ww).

        `tap_decomp` = how many 2-D CONV_2D taps a causal 3-D conv needs
        (kT when the temporal taps are separable, else 1 for pure 2-D).
        """
        key = prefix + ".weight"
        if key not in w:
            return f, h, ww, 0
        shp = w[key]["shape"]
        co, ci = shp[0], shp[1]
        out_pos = f * h * ww
        macs = conv_macs(shp, out_pos)
        record("conv", prefix, out_pos, macs, ci, co, shp[2:], nin_chunks=tap_decomp,
               fhw=(f, h, ww))
        return f, h, ww, macs

    total = 0

    # ---- conv_in: z_dim -> base_dim, causal 3-D -------------------------------
    f, h, ww, m = do_conv("decoder.conv_in", F, H, W, tap_decomp=3)
    total += m
    print(f"conv_in                 out {f}x{h}x{ww}   {m/1e9:8.2f} GMAC")

    # ---- mid_block: full 3-D self-attention over F*H*W tokens ----------------
    nt = F * H * W
    c = 384
    attn_qkv = 1152 * c * nt * 1 * 1 * 1        # 1x1x1 conv == linear
    attn_sc = nt * nt * c                       # QK^T  (fp16 score block on device)
    attn_pv = nt * nt * c                       # softmax x V
    attn_pr = c * c * nt
    attn_m = attn_qkv + attn_sc + attn_pv + attn_pr
    total += attn_m
    record("attention", "decoder.mid_block.attentions.0", nt, attn_m, c, c, None)
    print(f"mid attention (3-D)     {nt} tokens     {attn_m/1e9:8.2f} GMAC")

    # mid_block resnets: 2 x (conv1, conv2) @ same resolution
    for r in range(2):
        for cv in ("conv1", "conv2"):
            _, _, _, m = do_conv(f"decoder.mid_block.resnets.{r}.{cv}", f, h, ww, 3)
            total += m
    print(f"mid_block (2 resnets)                                   "
          f"{sum(o['macs'] for o in ops if 'mid_block.resnets' in o['name'])/1e9:8.2f} GMAC")

    # ---- up_blocks: 4 levels, reversed dim_mult = 4,4,2,1 --------------------
    n_res = 3   # = num_res_blocks + 1 for the decoder's deep blocks (read off the tensors)
    for ub in range(4):
        for r in range(n_res):
            for cv in ("conv1", "conv2", "conv_shortcut"):
                _, _, _, m = do_conv(f"decoder.up_blocks.{ub}.resnets.{r}.{cv}", f, h, ww, 3)
                total += m
        up = f"decoder.up_blocks.{ub}.upsamplers.0"
        if up + ".time_conv.weight" in w:                 # temporal 2x: F' = 2F-1
            shp = w[up + ".time_conv.weight"]["shape"]
            m = conv_macs(shp, f * h * ww)                # kT=3 taps at F_in
            total += m
            record("conv", up + ".time_conv", f * h * ww, m, shp[1], shp[0], shp[2:],
                   nin_chunks=3, fhw=(f, h, ww))
            f = 2 * f - 1
        if up + ".resample.1.weight" in w:                # spatial 2x then 3x3 conv
            h, ww = 2 * h, 2 * ww
            _, _, _, m = do_conv(up + ".resample.1", f, h, ww, 1)
            total += m
        print(f"after up_block {ub}      out {f}x{h}x{ww}   cum {total/1e9:8.2f} GMAC")

    # ---- head ---------------------------------------------------------------
    _, _, _, m = do_conv("decoder.conv_out", f, h, ww, 3)
    total += m
    print(f"conv_out                out {f}x{h}x{ww}   {m/1e9:8.2f} GMAC")

    conv_m = sum(o["macs"] for o in ops if o["kind"] == "conv")
    print()
    print(f"output frames           {4*p['F_z']-3} (expect {p['frames']})   "
          f"{'OK' if 4*p['F_z']-3 == p['frames'] else 'MISMATCH'}")
    print(f"output resolution       {h}x{ww} (expect {p['H']}x{p['W']})   "
          f"{'OK' if (h, ww) == (p['H'], p['W']) else 'MISMATCH'}")
    print(f"ops                     {len(ops)}  ({sum(1 for o in ops if o['kind']=='conv')} convs)")
    print(f"conv MACs               {conv_m/1e9:.2f} GMAC  ({100*conv_m/total:.1f} % of total)")
    print(f"TOTAL MACs              {total/1e9:.2f} GMAC  = {2*total/1e12:.3f} TFLOP")
    print()
    print("conv shape classes (unit of measurement on the device):")
    print(f"  {'class':58s} {'ops':>3s} {'MACs/op':>10s} {'total GMAC':>11s}")
    for k, v in sorted(classes.items(), key=lambda kv: -kv[1]["macs"]):
        print(f"  {k:58s} {v['ops']:3d} {v['macs']/1e9:10.3f} {v['ops']*v['macs']/1e9:11.2f}")

    if args.json:
        out = {
            "preset": "preset_a", "frames": p["frames"], "latent": [F, H, W],
            "output_fhw": [f, h, ww],
            "total_macs": total, "conv_macs": conv_m,
            "attention_macs": attn_m,
            "ops": ops,
            "classes": {k: {**v, "total_macs": v["ops"] * v["macs"]}
                        for k, v in classes.items()},
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
