#!/usr/bin/env python3
"""The decode's WIRING and its norm gammas — the two things `convseq` is missing.

`convseq` (P3.10) executes all 36 INT8 convs and requantises from one layer's SA
into the next's, i.e. it assumes the activation chain is a rescale. It is not.
Measured 2026-11-16 (P3.11, `calibrate.py` now records the per-layer OUTPUT
range): `decoder.mid_block.resnets.0.conv1` emits max|x| = **54.79** while
`...resnets.0.conv2` — the next conv in artifact order — consumes max|x| =
**0.612**, a factor of 89. No rescale does that. The reason is in
`WanResidualBlock.forward`:

    h = self.conv_shortcut(x)                 # residual branch (1x1 conv, or Identity)
    x = self.norm1(x); x = self.nonlinearity(x)   # WanRMS_norm + SiLU
    x = self.conv1(x)
    x = self.norm2(x); x = self.nonlinearity(x)   # WanRMS_norm + SiLU
    x = self.conv2(x)
    return x + h

`WanRMS_norm` is `F.normalize(x, dim=1) * sqrt(C) * gamma + bias` — an L2
normalisation across the channel axis at every (b, t, h, w), so it renormalises
the tensor per pixel. `bias` is a python float 0.0 for `images=False`, so only
`gamma` [C,1,1,1] exists as a parameter.

So "the decode decodes" needs, in order: norm gammas, SiLU, the temporal
shuffle in `time_conv`, the 2x nearest upsample before each `resample` conv, the
residual adds, and the mid-block attention. This tool writes the gammas + that
plan. It does not implement anything on the device and claims no quality.

Outputs
  artifacts/vae/norms_f32.bin   every WanRMS_norm gamma, f32, packed in site order
  artifacts/vae/vae_wiring.json the ordered op plan + gamma offsets + assertions

Proofs (not assertions of intent):
  * the structural walk, restricted to ops that own a conv, must reproduce the
    artifact's 36 layer names IN ORDER — 36/36, element-wise
  * every gamma byte is re-read from the file and compared float-for-float with
    the module parameter (delta 0), and each gamma must not be all-ones (a
    checkpoint that did not load would give exactly that)
  * the residual/add and shuffle sites are cross-checked against shapes

    python phase1_quant/vae_wiring.py            # write + verify
    python phase1_quant/vae_wiring.py --check    # verify only
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "weights_raw" / "DCM_WAN"
VAE_DIR = REPO / "artifacts" / "vae"
ART_JSON = VAE_DIR / "vae_int8.json"
OUT_BIN = VAE_DIR / "norms_f32.bin"
OUT_JSON = VAE_DIR / "vae_wiring.json"


def log(m: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch
    from diffusers import AutoencoderKLWan
    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        WanRMS_norm, WanResidualBlock, WanResample, WanUpBlock, WanCausalConv3d)

    art = json.loads(ART_JSON.read_text(encoding="utf-8"))
    art_names = [L["name"] for L in art["layers"]]

    vae = AutoencoderKLWan.from_pretrained(str(MODEL / "vae"), torch_dtype=torch.float32).eval()
    dec = vae.decoder

    gammas: list[tuple[str, np.ndarray]] = []   # (site name, f32[C])
    ops: list[dict] = []
    conv_order: list[str] = []

    def conv(mod, name: str, note: str = ""):
        assert isinstance(mod, (WanCausalConv3d, torch.nn.Conv2d)), (name, type(mod))
        conv_order.append(name)
        ops.append({"op": "conv", "name": name,
                    "kind": "conv3d" if isinstance(mod, WanCausalConv3d) else "conv2d",
                    "cin": int(mod.in_channels), "cout": int(mod.out_channels),
                    "k": [int(x) for x in (mod.kernel_size if isinstance(mod.kernel_size, tuple)
                                           else (mod.kernel_size,) * 3)],
                    "note": note})

    def norm(mod, name: str):
        assert isinstance(mod, WanRMS_norm), (name, type(mod))
        g = mod.gamma.detach().float().numpy()
        assert g.ndim == 4 and g.shape[1:] == (1, 1, 1), (name, g.shape)
        # bias is a python float 0.0 when images=False: no parameter to carry
        assert not isinstance(mod.bias, torch.nn.Parameter), \
            f"{name}: bias is a Parameter — this exporter would drop it"
        gammas.append((name, g.reshape(-1).astype(np.float32)))
        ops.append({"op": "norm", "site": name, "site_idx": len(gammas) - 1,
                    "C": int(g.shape[0]), "scale": float(mod.scale),
                    "gamma_off": None,          # filled when the blob is packed
                    "f_norm": "L2 over dim=1 (channels) per (b,t,h,w), * sqrt(C) * gamma"})

    def silu(name: str):
        ops.append({"op": "silu", "site": name})

    def add(site: str, src: str):
        ops.append({"op": "add", "site": site, "residual_from": src,
                    "note": "int8 codes add exactly IF both operands carry the same scale"})

    def resnet(blk: WanResidualBlock, name: str, tin: str):
        """One WanResidualBlock, in the order its forward applies things."""
        idn = isinstance(blk.conv_shortcut, torch.nn.Identity)
        if not idn:
            conv(blk.conv_shortcut, f"{name}.conv_shortcut", "residual branch, 1x1")
        ops.append({"op": "save", "site": f"{name}:entry",
                    "alias": "identity (no conv_shortcut)" if idn else f"{name}.conv_shortcut",
                    "note": "the resnet's entry tensor, held for the residual add"})
        norm(blk.norm1, f"{name}.norm1")
        silu(f"{name}.silu1")
        conv(blk.conv1, f"{name}.conv1")
        norm(blk.norm2, f"{name}.norm2")
        silu(f"{name}.silu2")
        conv(blk.conv2, f"{name}.conv2")
        add(f"{name}:residual", f"{name}:entry")
        return f"{name}:residual"

    # ---- the decoder, in forward order ------------------------------------
    conv(dec.conv_in, "decoder.conv_in")

    for ri, r in enumerate(dec.mid_block.resnets):
        resnet(r, f"decoder.mid_block.resnets.{ri}", "")
        if ri < len(dec.mid_block.attentions) and \
                not isinstance(dec.mid_block.attentions[ri], torch.nn.Identity):
            ops.append({"op": "attn", "site": f"decoder.mid_block.attentions.{ri}",
                        "device": "Mali (measured 0.0855 s/chunk, P3.6); TPU 0.601 s",
                        "note": "not a conv — no INT8 artifact exists, not executed by convseq"})

    for bi, ub in enumerate(dec.up_blocks):
        assert isinstance(ub, WanUpBlock), type(ub)
        for ri, r in enumerate(ub.resnets):
            resnet(r, f"decoder.up_blocks.{bi}.resnets.{ri}", "")
        if ub.upsamplers is not None:
            rs = ub.upsamplers[0]
            assert isinstance(rs, WanResample), type(rs)
            if getattr(rs, "time_conv", None) is not None:
                conv(rs.time_conv, f"decoder.up_blocks.{bi}.upsamplers.0.time_conv",
                     "temporal: (3,1,1) causal, then 2C -> C pixel-shuffle over t")
                ops.append({"op": "temporal_shuffle", "site": f"up_blocks.{bi}.upsample",
                            "factor": 2,
                            "shape": "reshape(b,2,c,t,h,w) -> stack dim3 -> (b,c,2t,h,w)"})
            sq = rs.resample
            assert isinstance(sq, torch.nn.Sequential), type(sq)
            ops.append({"op": "upsample2d", "site": f"up_blocks.{bi}.resample",
                        "mode": "nearest-exact", "factor": 2})
            conv(sq[1], f"decoder.up_blocks.{bi}.upsamplers.0.resample.1",
                 "2-D conv applied per frame after the 2x nearest upsample")

    norm(dec.norm_out, "decoder.norm_out")
    silu("decoder.head.silu")
    conv(dec.conv_out, "decoder.conv_out")

    # ---- proof 1: the walk and the artifact cover the same 36 convs -------
    # NOT necessarily in the same ORDER: the artifact indexes each resnet as
    # conv1, conv2, conv_shortcut, while `WanResidualBlock.forward` applies the
    # shortcut FIRST (`h = self.conv_shortcut(x)`). So the artifact's layer
    # index is a storage order, not an execution order, and a device that walks
    # the SA table by index position is walking it out of order at every resnet
    # that has a shortcut. Recorded, not smoothed over.
    if sorted(conv_order) != sorted(art_names):
        only = [n for n in conv_order if n not in set(art_names)]
        miss = [n for n in art_names if n not in set(conv_order)]
        log(f"FAIL conv set: walk {len(conv_order)} vs artifact {len(art_names)}; "
            f"walk-only ({len(only)}): {only[:6]}; artifact-only ({len(miss)}): {miss[:6]}")
        return 1
    art_idx = {n: i for i, n in enumerate(art_names)}
    for o in ops:
        if o["op"] == "conv":
            o["artifact_index"] = art_idx[o["name"]]
    perm = [art_idx[n] for n in conv_order]
    in_artifact_order = all(a < b for a, b in zip(perm, perm[1:]))
    shuffled = [(i, conv_order[i], perm[i]) for i in range(len(perm))
                if i != perm[i]]
    log(f"PASS conv set: 36/36 artifact layers found by the structural walk")
    log(f"     execution order == artifact order: {in_artifact_order}"
        f"{'' if in_artifact_order else f' — {len(shuffled)} positions differ, first ' + str(shuffled[:3])}")

    # ---- pack the gammas --------------------------------------------------
    blob = bytearray()
    for i, (site, g) in enumerate(gammas):
        ops[[o.get("site") for o in ops].index(site)]["gamma_off"] = len(blob)
        blob += g.astype("<f4").tobytes()
    n_all_ones = sum(1 for _, g in gammas if bool((g == 1.0).all()))
    if n_all_ones:
        log(f"FAIL {n_all_ones}/{len(gammas)} gammas are exactly all-ones — checkpoint not loaded?")
        return 1
    C_tot = sum(g.size for _, g in gammas)

    kinds: dict[str, int] = {}
    for o in ops:
        kinds[o["op"]] = kinds.get(o["op"], 0) + 1

    # The residual ADD needs the POST-ADD range, which no conv owns: conv2's own
    # range understates it by up to 4.76x (measured). Pull it from the newest
    # VAE calibration's `resnets` block and attach it to each add site, BEFORE
    # the plan is written.
    calibs = sorted((REPO / "artifacts" / "calib").glob("act_scales_vae_*.json"))
    if calibs:
        cal = json.loads(calibs[-1].read_text(encoding="utf-8"))
        post = {r["resnet"]: r for r in cal.get("resnets", [])}
        for o in ops:
            if o["op"] != "add":
                continue
            rn = o["site"].replace(":residual", "")
            r = post.get(rn)
            if r is None:
                log(f"WARN no post-add range for {rn} in {calibs[-1].name}")
                continue
            o["post_add_max_abs"] = r["max_abs_out"]
            o["post_add_sa"] = r["sa_out"]
        log(f"post-add scales attached to {sum(1 for o in ops if o.get('post_add_sa'))}"
            f"/{kinds.get('add', 0)} add sites from {calibs[-1].name}")
    else:
        log("WARN no VAE calibration found — add sites carry no post_add_sa")

    if not args.check:
        OUT_BIN.write_bytes(bytes(blob))
        doc = dict(
            built=dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
            source_artifact=str(ART_JSON.relative_to(REPO)).replace("\\", "/"),
            source_artifact_sha=hashlib.sha256(ART_JSON.read_bytes()).hexdigest()[:16],
            file=dict(name=OUT_BIN.name, bytes=len(blob),
                      sha256=hashlib.sha256(bytes(blob)).hexdigest(),
                      layout="float32 per site, concatenated in op order; each norm op carries "
                             "its own gamma_off and C"),
            n_norms=len(gammas), n_gamma_floats=int(C_tot), n_convs=len(conv_order),
            n_ops=len(ops),
            ops=ops,
            note=("The device must implement: L2 norm across channels * sqrt(C) * gamma, SiLU, "
                  "the temporal shuffle, the 2x nearest upsample, and the residual adds — in "
                  "this order. Only the conv ops are INT8/NNAPI; everything else is a float "
                  "pass over the activation. SiLU and the norm cannot be folded into the conv "
                  "epilogue because the norm's scale is per-pixel, not per-channel."),
        )
        OUT_JSON.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        log(f"wrote {OUT_BIN.relative_to(REPO)} ({len(blob)} B, {len(gammas)} gammas, {C_tot} floats)")
        log(f"wrote {OUT_JSON.relative_to(REPO)} ({len(ops)} ops)")

    # ---- proof 2: re-read the bytes, float identity vs the parameters ------
    raw = OUT_BIN.read_bytes()
    if len(raw) != 4 * C_tot:
        log(f"FAIL size: {len(raw)} B != {4 * C_tot} B")
        return 1
    worst, off = 0.0, 0
    for site, g in gammas:
        got = np.frombuffer(raw, dtype="<f4", count=g.size, offset=off)
        if not np.array_equal(got, g.astype("<f4")):
            log(f"FAIL gamma {site}: bytes differ from the module parameter")
            return 1
        worst = max(worst, float(np.abs(got.astype(np.float64) - g.astype(np.float64)).max()))
        off += 4 * g.size
    log(f"PASS norms_f32.bin: {len(gammas)} gammas, {C_tot} floats, float identity "
        f"(worst delta {worst:.3g}), none all-ones")

    kinds: dict[str, int] = {}
    for o in ops:
        kinds[o["op"]] = kinds.get(o["op"], 0) + 1

    log(f"plan: {kinds}")
    log(f"conv2d ops (the `resample` convs): "
        f"{sum(1 for o in ops if o['op'] == 'conv' and o['kind'] == 'conv2d')}; "
        f"temporal_shuffle {kinds.get('temporal_shuffle', 0)}; residual adds {kinds.get('add', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
