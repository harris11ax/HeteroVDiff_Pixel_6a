#!/usr/bin/env python3
"""P3.13 — does the device's WIRED decode *decode*? Device codes vs the fp32 reference.

P3.12 proved the walk executes (592/592, saturation 0.0721 %) and said nothing about
whether its output is a decode. This closes that: `npu_block convwire ... 1` now writes
the final activation codes and the latent codes the first conv actually consumed, and
this script dequantises both and compares the device's decode against the fp32 reference
**from the identical quantised latent**.

Three references, because they separate three different things:

  A  fp32 weights, attention ON            — the decode as the model defines it
  B  fp32 weights, attention SKIPPED       — the program the device runs (OP SKIP attn)
  C  INT8 weights + INT8 activations at the measured SA, attention skipped
                                           — the SAME quantisation the device applies,
                                             simulated on the host

  device vs A = the total cost of running this decode on the Edge TPU
  device vs B = the same, with the attention skip removed from the error
  device vs C = what is left after the scheme is matched (HAL numerics, per-tap
                splitting, the scalar glue) — the device-fidelity number

Guards, so the comparison cannot silently be about two different tensors:
  * the dumped latent codes are re-derived from the device's own LCG (bit-exact) and its
    scale is asserted to equal `act_scales2_f32.bin[0]`, i.e. the table the host uses
  * the dump's shape/scale come from the device's sidecar, never from an assumption here
  * config C asserts every act-scale hook fired (36/36), as psnr_gate1 does

    .venv-ref/Scripts/python.exe tools/npu/vae_psnr.py --dump artifacts/vae/dump_<ts>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "phase1_quant"))
import psnr_gate1 as G1  # noqa: E402  load_artifact / rebuild_layer / psnr / ssim

ART = REPO / "artifacts" / "vae"
VAE_DIR = REPO / "weights_raw" / "DCM_WAN" / "vae"
OUT = REPO / "docs" / "device"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_meta(p: Path) -> dict:
    d: dict = {}
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


def load_codes(binp: Path, meta: dict) -> np.ndarray:
    """int8 file in the harness's [f][h][w][c] order -> [f, c, h, w] float at its scale."""
    raw = np.fromfile(binp, dtype=np.int8)
    want = meta["f"] * meta["c"] * meta["h"] * meta["w"]
    if raw.size != want:
        raise SystemExit(f"{binp.name}: {raw.size} codes vs meta {want} "
                         f"({meta['f']}x{meta['c']}x{meta['h']}x{meta['w']})")
    a = raw.reshape(meta["f"], meta["h"], meta["w"], meta["c"]).transpose(0, 3, 1, 2)
    return np.ascontiguousarray(a).astype(np.float32) * float(meta["sa"])


def lcg_codes(n: int) -> np.ndarray:
    """The device's synthetic latent (xorshift-free LCG, byte = (x>>24)&0xFF), re-derived."""
    out = np.empty(n, dtype=np.uint8)
    x = np.uint32(0x9E3779B9)
    for i in range(n):
        x = np.uint32((int(x) * 1664525 + 1013904223) & 0xFFFFFFFF)
        out[i] = np.uint8((int(x) >> 24) & 0xFF)
    return out.view(np.int8)


def metrics(ref: torch.Tensor, dev: torch.Tensor) -> dict:
    d = (dev - ref)
    rmse = float(torch.sqrt(torch.mean(d ** 2)))
    rms_ref = float(torch.sqrt(torch.mean(ref ** 2)))
    peak = float(ref.abs().max())
    cos = float(torch.nn.functional.cosine_similarity(dev.flatten(), ref.flatten(), dim=0))
    return {
        "psnr_db": G1.psnr(ref, dev),                     # data_range 2.0, as Gate 1
        "psnr_db_datapeak": (float("inf") if rmse == 0 else
                             20.0 * math.log10(peak / rmse)) if peak > 0 else None,
        "ssim": G1.ssim(ref, dev),
        "cosine": cos,
        "rmse": rmse,
        "rel_rmse": rmse / max(rms_ref, 1e-12),
        "max_abs": float(d.abs().max()),
        "ref_absmax": peak,
        "ref_rms": rms_ref,
        "per_frame_psnr": [G1.psnr(ref[:, :, f], dev[:, :, f]) for f in range(ref.shape[2])],
        "per_frame_rel_rmse": [
            float(torch.sqrt(torch.mean((dev[:, :, f] - ref[:, :, f]) ** 2)))
            / max(float(torch.sqrt(torch.mean(ref[:, :, f] ** 2))), 1e-12)
            for f in range(ref.shape[2])],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None, help="artifacts/vae/dump_<ts> directory")
    ap.add_argument("--capture", default=None, help="the vae_wire_<ts>.json this dump came from")
    ap.add_argument("--out", default="vae_psnr")
    ap.add_argument("--skip-c", action="store_true", help="skip the INT8-simulation reference")
    args = ap.parse_args()

    ddir = Path(args.dump) if args.dump else sorted(ART.glob("dump_*"))[-1]
    if not (ddir / "vae_dump.bin").exists():
        raise SystemExit(f"no dump in {ddir} — run `vae_wire.py --run --dump` first")
    dmeta = read_meta(ddir / "vae_dump.txt")
    lmeta = read_meta(ddir / "vae_latent.txt")
    log(f"dump {ddir.name}: out {dmeta['f']}x{dmeta['c']}x{dmeta['h']}x{dmeta['w']} "
        f"sa={dmeta['sa']:.6g} | latent {lmeta['f']}x{lmeta['c']}x{lmeta['h']}x{lmeta['w']} "
        f"sa={lmeta['sa']:.6g}")

    # ---- the device's own claims, re-derived (not trusted)
    lat_codes = np.fromfile(ddir / "vae_latent.bin", dtype=np.int8)
    lcg = lcg_codes(lat_codes.size)
    lcg_ident = bool(np.array_equal(lat_codes, lcg))
    sa2 = np.fromfile(ART / "act_scales2_f32.bin", dtype="<f4")
    nlay = sa2.size // 2
    sa_in0 = float(sa2[0])
    log(f"latent: {lat_codes.size:,} codes, LCG re-derivation bit-identical = {lcg_ident}; "
        f"device latent scale {lmeta['sa']:.9g} vs table SA_in[0] {sa_in0:.9g}")
    if not lcg_ident:
        raise SystemExit("the dumped latent is not the LCG the harness declares — refusing to compare")
    if abs(float(lmeta["sa"]) - sa_in0) > 1e-9 * max(1.0, sa_in0):
        raise SystemExit(f"device latent scale {lmeta['sa']} != table SA_in[0] {sa_in0} — the "
                         f"walk and the host would be quantising different tensors")

    dev_np = load_codes(ddir / "vae_dump.bin", dmeta)          # [f, c, h, w]
    z_np = load_codes(ddir / "vae_latent.bin", lmeta)          # [f, c, h, w]
    # the harness stores [f][h][w][c]; torch video tensors are [B, C, F, H, W] — so the
    # frame axis moves inside the channel axis, it does not become the batch
    z = torch.from_numpy(z_np).permute(1, 0, 2, 3).unsqueeze(0)
    dev = torch.from_numpy(dev_np).permute(1, 0, 2, 3).unsqueeze(0)
    log(f"device decode: {tuple(dev.shape)} range {float(dev.min()):.4f}..{float(dev.max()):.4f} "
        f"(codes at sa={float(dmeta['sa']):.6g})")
    log(f"latent:         {tuple(z.shape)} range {float(z.min()):.4f}..{float(z.max()):.4f} "
        f"rms {float(torch.sqrt(torch.mean(z**2))):.4f}")

    # ---- the reference model
    from diffusers import AutoencoderKLWan

    man, wbytes, scales = G1.load_artifact()
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(str(VAE_DIR), torch_dtype=torch.float32).eval()
    params = dict(vae.named_parameters())
    log(f"reference model loaded in {time.time() - t0:.1f} s")

    rebuilt: dict[str, torch.Tensor] = {}
    for L in man["layers"]:
        rebuilt[L["weight_tensor"]] = torch.from_numpy(G1.rebuild_layer(L, wbytes, scales))
    original = {n: params[n].detach().clone() for n in rebuilt}
    log(f"artifact: {len(rebuilt)} decoder weights rebuilt from INT8")

    act_sa = {}
    # vae_act_scales.json is a manifest (P3.10), not the table — follow its pointer.
    cands = []
    man_p = ART / "vae_act_scales.json"
    if man_p.exists():
        m = json.loads(man_p.read_text(encoding="utf-8"))
        if m.get("source_calib"):
            cands.append(REPO / m["source_calib"] if not Path(m["source_calib"]).is_absolute()
                         else Path(m["source_calib"]))
    cands += sorted((ART.parent / "calib").glob("act_scales_vae_*.json"))
    for p in reversed([c for c in cands if Path(c).exists()]):
        try:
            rows = json.loads(Path(p).read_text(encoding="utf-8"))["layers"]
        except (KeyError, json.JSONDecodeError):
            continue
        got = {r["layer"]: float(r["sa"]) for r in rows if r.get("sa")}
        if len(got) == 36:
            act_sa = got
            log(f"act scales from {Path(p).name}")
            break
    have = {L["name"] for L in man["layers"]}
    if act_sa and set(act_sa) != have:
        raise SystemExit(f"act scales cover {len(act_sa)} layers, artifact has {len(have)}")
    log(f"act scales: {len(act_sa)}/{len(have)} layers "
        f"(SA {min(act_sa.values()):.6g}..{max(act_sa.values()):.6g})" if act_sa
        else "act scales: NONE — the INT8-activation config will be skipped")

    attn_mod = vae.decoder.mid_block.attentions[0]

    def skip_attn():
        """OP SKIP attn: the block's own body runs, its output is discarded."""
        return attn_mod.register_forward_hook(lambda m, inp, out: inp[0])

    def skip_pqc():
        """`post_quant_conv` is NOT in the artifact (36 layers = decoder only) and the
        program has no op for it, so the device's `decoder.conv_in` consumes the latent
        codes directly. The reference must do the same, or the comparison is between two
        different inputs — a 16->16 channel mixing matrix, which is what turned a first
        attempt into a 0.46-cosine 'failure'. Reported as a deviation, not folded."""
        return vae.post_quant_conv.register_forward_hook(lambda m, inp, out: inp[0])

    def act_hooks():
        hs, fired = [], set()

        def mk(name):
            sa = act_sa[name]

            def hook(_m, inp):
                x = inp[0]
                fired.add(name)
                return (torch.clamp(torch.round(x / sa), -127.0, 127.0) * sa,) + tuple(inp[1:])
            return hook

        for name in act_sa:
            hs.append(vae.get_submodule(name).register_forward_pre_hook(mk(name)))
        return hs, fired

    configs = [("A: fp32, attention ON, post_quant_conv bypassed (the program's input path)",
                {"no_pqc": 1}, {}),
               ("B: fp32, attention SKIPPED (the program the device runs)",
                {"no_pqc": 1, "skip_attn": 1}, {})]
    if act_sa and not args.skip_c:
        configs.append(("C: INT8 weights + INT8 activations at measured SA, attention skipped",
                        {"no_pqc": 1, "skip_attn": 1, "int8_w": 1, "int8_a": 1}, {}))

    results = []
    for label, flags, meta in configs:
        with torch.no_grad():
            for n, t in original.items():
                params[n].copy_(t)
            if flags.get("int8_w"):
                for n, t in rebuilt.items():
                    params[n].copy_(t)
        hooks = []
        if flags.get("no_pqc"):
            hooks.append(skip_pqc())
        if flags.get("skip_attn"):
            hooks.append(skip_attn())
        fired = set()
        if flags.get("int8_a"):
            hs, fired = act_hooks()
            hooks += hs
        t0 = time.time()
        with torch.no_grad():
            ref = vae.decode(z).sample.clone()
        dt_s = time.time() - t0
        for h in hooks:
            h.remove()
        if flags.get("int8_a") and len(fired) != len(act_sa):
            raise SystemExit(f"act hooks fired {len(fired)}/{len(act_sa)} — the decode did not "
                             f"run the path these scales describe")
        m = metrics(ref, dev)
        results.append({"config": label, **meta, **m, "decode_s": dt_s})
        log(f"{label}: rel-RMSE {m['rel_rmse'] * 100:.2f} % | cosine {m['cosine']:.6f} | "
            f"PSNR(2.0) {m['psnr_db']:.2f} dB | PSNR(peak {m['ref_absmax']:.3g}) "
            f"{m['psnr_db_datapeak']:.2f} dB | SSIM {m['ssim']:.4f}")

    best = max(results, key=lambda r: r["cosine"])
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    doc = {
        "kind": "vae_device_psnr",
        "timestamp": ts,
        "dump_dir": str(ddir),
        "dump_meta": {"output": dmeta, "latent": lmeta},
        "latent_check": {"lcg_bit_identical": lcg_ident, "codes": int(lat_codes.size),
                         "device_sa": float(lmeta["sa"]), "table_sa_in0": sa_in0},
        "device_output": {"shape": list(dev.shape), "min": float(dev.min()),
                          "max": float(dev.max()), "rms": float(torch.sqrt(torch.mean(dev ** 2)))},
        "references": results,
        "closest_reference": best["config"],
        "note": "device numbers are the dumped int8 codes at the device's own declared scale; "
                "no modelled value enters this file.",
    }
    js = OUT / f"{args.out}_{ts}.json"
    js.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    md = [f"# VAE decode on the device vs the fp32 reference — {ts}", "",
          f"dump `{ddir.name}` · output {dmeta['f']}×{dmeta['c']}×{dmeta['h']}×{dmeta['w']} "
          f"at sa **{float(dmeta['sa']):.6g}** · latent {lmeta['f']}×{lmeta['c']}×"
          f"{lmeta['h']}×{lmeta['w']} at sa {float(lmeta['sa']):.6g}", "",
          "Same quantised latent on both sides: the device's own latent codes, re-derived from "
          "its LCG (bit-identical) and dequantised at the table's `SA_in[0]`.", "",
          "| reference | rel-RMSE | cosine | PSNR (2.0) | PSNR (data peak) | SSIM |",
          "|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['config']} | {r['rel_rmse'] * 100:.2f} % | {r['cosine']:.6f} | "
                  f"{r['psnr_db']:.2f} dB | {r['psnr_db_datapeak']:.2f} dB | {r['ssim']:.4f} |")
    md += ["", f"Closest reference by cosine: **{best['config']}**.", "",
           "## Per frame (vs " + results[0]["config"] + ")", "",
           "| frame | PSNR dB | rel-RMSE |", "|---|---|---|"]
    for f, (p, r_) in enumerate(zip(results[0]["per_frame_psnr"], results[0]["per_frame_rel_rmse"])):
        md.append(f"| {f} | {p:.2f} | {r_ * 100:.2f} % |")
    md.append("")
    mdp = OUT / f"{args.out}_{ts}.md"
    mdp.write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {js}\nwrote {mdp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
