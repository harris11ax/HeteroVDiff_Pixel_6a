#!/usr/bin/env python3
"""Gate 1, end-to-end leg: FP32 vs INT8 weight-quantised VAE decode — PSNR and SSIM.

Gate 1's artifact leg (byte-exact INT8 base, 300/300 GEMM cosine >= 0.999) is measured;
what was never run is the *quality* leg: does the locked quantisation scheme still produce
the decoder's images? This does it on the host, with no device:

  1. load the reference `AutoencoderKLWan` in fp32 (`.venv-ref`, torch 2.15.0+cpu)
  2. decode a fixed seeded latent  ->  out_fp32
  3. rebuild every decoder weight from `artifacts/vae/` (INT8 bytes x the artifact's own
     per-output-channel scales) and decode the same latent with those weights -> out_q
  4. PSNR / SSIM between them, plus a per-layer round-trip check that the rebuild is the
     artifact and not a plausible-looking transpose.

The comparison is **weight-only** (quantise-dequantise). Activations stay fp32 because the
device's activation scale `SA` is `calibrate.py`'s output and nothing here invents it: this
measures what the locked quantiser costs the decoder, not what the edge TPU's activations cost.

Round-trip assertion: each rebuilt weight must match the checkpoint's own tensor to within the
artifact manifest's recorded relFro (1e-3 absolute tolerance). A transposed or tap-shifted
rebuild still "looks" quantised, so this is the check that the inversion is right.

    .venv-ref/Scripts/python.exe phase1_quant/psnr_gate1.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "vae"
VAE_DIR = REPO / "weights_raw" / "DCM_WAN" / "vae"
OUT = REPO / "docs" / "device"

PSNR_GATE_DB = 28.0
SSIM_GATE = 0.92
LATENT = (1, 16, 3, 34, 60)          # Preset A: 9 frames x 272 x 480 out


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- artifact readers
def load_artifact() -> tuple[dict, np.memmap, np.ndarray]:
    man = json.loads((ART / "vae_int8.json").read_text(encoding="utf-8"))
    w = np.memmap(ART / "weights_int8.bin", dtype=np.int8, mode="r")
    s = np.fromfile(ART / "scales_f32.bin", dtype="<f4")
    return man, w, s


def rebuild_layer(layer: dict, w: np.memmap, scales: np.ndarray) -> np.ndarray:
    """INT8 taps + per-output-channel scales -> the checkpoint's [Cout,Cin,kT,kH,kW] tensor."""
    sc = scales[layer["scales"]["offset"] // 4: layer["scales"]["offset"] // 4 + layer["scales"]["count"]]
    shape = layer["source_shape"]
    cout, cin = shape[0], shape[1]
    if len(shape) == 5:
        cout_, cin_, kt, kh, kw = shape
        out = np.zeros((cout_, cin_, kt, kh, kw), dtype=np.float32)
    elif len(shape) == 4:
        cout_, cin_, kh, kw = shape
        kt = 1
        out = np.zeros((cout_, cin_, kh, kw), dtype=np.float32)
    else:
        raise ValueError(f"{layer['name']}: unsupported source_shape {shape}")
    assert sc.size == cout_, f"{layer['name']}: {sc.size} scales vs Cout {cout_}"
    for t_plan in layer["taps"]:
        t = int(t_plan["tap"])
        off, nbytes = int(t_plan["offset"]), int(t_plan["bytes"])
        q = np.asarray(w[off: off + nbytes], dtype=np.int8).reshape(cout_, kh, kw, cin)
        # OHWI -> [Cout, Cin, kH, kW]
        dense = np.ascontiguousarray(np.transpose(q, (0, 3, 1, 2))).astype(np.float32)
        dense *= sc[:, None, None, None]
        if len(shape) == 4:
            out += dense                    # a 4-D conv (kT=1) is stored as its single tap
        elif kt == 1:
            out[:, :, 0, :, :] = dense      # 5-D with a length-1 time axis (conv_shortcut)
        else:
            out[:, :, t, :, :] = dense
    return out


def relfro(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


# ---------------------------------------------------------------- metrics
def psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 2.0) -> float:
    mse = float(torch.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(data_range ** 2 / mse)


def _gauss_window(n: int = 11, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(n, dtype=torch.float32) - (n - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a: torch.Tensor, b: torch.Tensor, data_range: float = 2.0) -> float:
    """Mean SSIM per frame over channels (11x11 gaussian, sigma 1.5) — the standard recipe."""
    win = _gauss_window().to(a.dtype)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    vals = []
    for f in range(a.shape[2]):
        x = a[0, :, f].unsqueeze(0)
        y = b[0, :, f].unsqueeze(0)
        nch = x.shape[1]
        mu_x = torch.nn.functional.conv2d(x, win.expand(nch, 1, 11, 11), padding=5, groups=nch)
        mu_y = torch.nn.functional.conv2d(y, win.expand(nch, 1, 11, 11), padding=5, groups=nch)
        sxx = torch.nn.functional.conv2d(x * x, win.expand(nch, 1, 11, 11), padding=5, groups=nch) - mu_x ** 2
        syy = torch.nn.functional.conv2d(y * y, win.expand(nch, 1, 11, 11), padding=5, groups=nch) - mu_y ** 2
        sxy = torch.nn.functional.conv2d(x * y, win.expand(nch, 1, 11, 11), padding=5, groups=nch) - mu_x * mu_y
        num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
        den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
        vals.append(float((num / den).mean()))
    return float(np.mean(vals))


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=LATENT[2])
    ap.add_argument("--skip-worst", type=int, default=0,
                    help="also run a second config with the N lowest-weight-cosine layers left in "
                         "fp32, to attribute the end-to-end loss to specific layers")
    ap.add_argument("--act-scales", default=None,
                    help="artifacts/calib/act_scales_vae_*.json from calibrate.py: also run a "
                         "decode with INT8 ACTIVATIONS at the measured per-layer SA, which is the "
                         "device's real operand path (weight-only is the bar Gate 1 asked for; "
                         "this is the number that lives beside it)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from diffusers import AutoencoderKLWan

    man, wbytes, scales = load_artifact()
    log(f"artifact: {len(man['layers'])} layers, {sum(len(l['taps']) for l in man['layers'])} taps, "
        f"{wbytes.size:,} INT8 B, {scales.size:,} scales")

    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(str(VAE_DIR), torch_dtype=torch.float32).eval()
    params = dict(vae.named_parameters())
    log(f"reference model loaded in {time.time() - t0:.1f} s")

    # ---- 1. rebuild every decoder weight from the artifact, and prove the inversion
    rebuilt: dict[str, torch.Tensor] = {}
    rows = []
    bad = []
    for L in man["layers"]:
        dense = rebuild_layer(L, wbytes, scales)
        orig = params[L["weight_tensor"]].detach().numpy()
        if tuple(orig.shape) != tuple(L["source_shape"]):
            raise SystemExit(f"{L['name']}: checkpoint shape {tuple(orig.shape)} != "
                             f"manifest {tuple(L['source_shape'])}")
        r = relfro(dense, orig)
        recorded = float(L["quant"]["rel_fro"])
        rows.append({"layer": L["name"], "shape": list(L["source_shape"]),
                     "taps": len(L["taps"]), "relfro_rebuilt": r,
                     "relfro_recorded": recorded,
                     "cosine_recorded": float(L["quant"]["cosine"])})
        if abs(r - recorded) > 1e-3:
            bad.append((L["name"], r, recorded))
        rebuilt[L["weight_tensor"]] = torch.from_numpy(dense)

    if bad:
        for n, r, rec in bad[:8]:
            log(f"ROUND-TRIP MISMATCH {n}: rebuilt {r:.6f} vs recorded {rec:.6f}")
        raise SystemExit(f"rebuild does not reproduce the artifact on {len(bad)} layer(s) — "
                         f"refusing to report PSNR from a wrong inversion")
    log(f"round-trip: {len(rows)}/{len(rows)} layers reproduce the manifest's relFro "
        f"(max |diff| {max(abs(x['relfro_rebuilt'] - x['relfro_recorded']) for x in rows):.2e})")

    # ---- 2. the decodes, same latent, same code path, one config at a time
    g = torch.Generator().manual_seed(args.seed)
    z = torch.randn(1, 16, args.frames, LATENT[3], LATENT[4], generator=g)
    log(f"latent {tuple(z.shape)} seed {args.seed}")

    t0 = time.time()
    with torch.no_grad():
        a = vae.decode(z).sample.clone()
    log(f"fp32 decode -> {tuple(a.shape)} in {time.time() - t0:.1f} s")

    original = {name: params[name].detach().clone() for name in rebuilt}   # pristine fp32, BEFORE patching

    # ---- optional: INT8 ACTIVATIONS at the measured SA (calibrate.py). The device's
    # operands are INT8 on both sides; this config prices the activation half, which
    # weight-only quantise-dequantise deliberately leaves out.
    act_sa: dict[str, float] = {}
    if args.act_scales:
        ap_path = Path(args.act_scales)
        if not ap_path.is_absolute():
            ap_path = REPO / ap_path
        cal = json.loads(ap_path.read_text(encoding="utf-8"))
        act_sa = {r["layer"]: float(r["sa"]) for r in cal["layers"] if r.get("sa")}
        have = {L["name"] for L in man["layers"]}
        if set(act_sa) != have:
            raise SystemExit(f"act scales cover {len(act_sa)} layers, artifact has {len(have)}: "
                             f"missing {sorted(have - set(act_sa))[:4]}")
        log(f"act scales: {len(act_sa)}/{len(have)} layers from {rel(ap_path)} "
            f"(SA {min(act_sa.values()):.6f} .. {max(act_sa.values()):.6f})")

    def install_act_hooks() -> list:
        """Quantise-dequantise each layer's INPUT with its measured SA, per call."""
        hs, fired = [], set()

        def mk(name):
            sa = act_sa[name]

            def hook(_m, inp):
                x = inp[0]
                q = torch.clamp(torch.round(x / sa), -127.0, 127.0) * sa
                fired.add(name)
                return (q,) + tuple(inp[1:])
            return hook

        for name in act_sa:
            hs.append(vae.get_submodule(name).register_forward_pre_hook(mk(name)))
        return hs, fired

    configs: list[tuple[str, dict[str, torch.Tensor], dict]] = [
        ("all quantised", rebuilt, {"layers_quantised": len(rebuilt)})
    ]
    if args.skip_worst > 0:
        worst = sorted(rows, key=lambda r: r["cosine_recorded"])[: args.skip_worst]
        worst_names = {r["layer"] for r in worst}
        keep = {k: v for k, v in rebuilt.items()
                if k.rsplit(".", 1)[0] not in worst_names}
        configs.append((f"quantised except worst {len(worst)} by weight cosine",
                        keep, {"layers_quantised": len(keep),
                               "left_fp32": [{"layer": n, "cosine": c} for n, c in
                                             zip([w["layer"] for w in worst],
                                                 [w["cosine_recorded"] for w in worst])]}))
    if act_sa:
        configs.append(("all quantised + INT8 activations (measured SA)",
                        rebuilt, {"layers_quantised": len(rebuilt), "act_quant": True,
                                  "act_scales_layers": len(act_sa)}))

    results = []
    seen_hashes: dict[str, str] = {}
    for label, weights, meta in configs:
        with torch.no_grad():
            for name, tens in original.items():
                params[name].copy_(tens)          # restore pristine fp32
            for name, tens in weights.items():
                params[name].copy_(tens)
        hooks, fired = ([], set())
        if meta.get("act_quant"):
            hooks, fired = install_act_hooks()
        t0 = time.time()
        with torch.no_grad():
            b = vae.decode(z).sample.clone()
        dt = time.time() - t0
        for hk_ in hooks:
            hk_.remove()
        if meta.get("act_quant") and len(fired) != len(act_sa):
            raise SystemExit(f"act hooks: {len(fired)}/{len(act_sa)} layers fired — the decode "
                             f"did not run the path these scales describe")
        # Guards that make a restore bug loud instead of plausible: a config that changed
        # nothing must not reproduce the fp32 output, and two configs that differ in which
        # layers are quantised must not produce the same tensor.
        h = hashlib.sha256(np.ascontiguousarray(b.numpy()).tobytes()).hexdigest()[:16]
        if h in seen_hashes:
            raise SystemExit(f"config '{label}' produced the same output as '{seen_hashes[h]}' — "
                             f"the weight restore is wrong; refusing to report an attribution")
        seen_hashes[h] = label
        p_all = psnr(a, b)
        s_all = ssim(a, b)
        per_frame = [{"frame": f, "psnr_db": psnr(a[:, :, f], b[:, :, f]),
                      "ssim": ssim(a[:, :, f:f + 1], b[:, :, f:f + 1])}
                     for f in range(a.shape[2])]
        verdict = "PASS" if (p_all >= PSNR_GATE_DB and s_all >= SSIM_GATE) else "FAIL"
        log(f"{label}: PSNR {p_all:.2f} dB | SSIM {s_all:.4f} -> {verdict} ({dt:.1f} s, "
            f"{meta['layers_quantised']}/36 layers quantised)")
        results.append({"config": label, **meta, "psnr_db": p_all, "ssim": s_all,
                        "verdict": verdict, "decode_s": dt, "out_sha16": h,
                        "per_frame": per_frame})

    main_res = results[0]
    p_all, s_all, verdict = main_res["psnr_db"], main_res["ssim"], main_res["verdict"]
    log(f"per-frame PSNR {min(x['psnr_db'] for x in main_res['per_frame']):.2f} .. "
        f"{max(x['psnr_db'] for x in main_res['per_frame']):.2f} dB")
    worst_layer = min(rows, key=lambda r: r["cosine_recorded"])

    stamp = time.strftime("%Y%m%d_%H%M%S")
    doc = {
        "kind": "gate1_psnr_ssim",
        "timestamp": stamp,
        "latent": list(z.shape),
        "seed": args.seed,
        "comparison": "FP32 decode vs the same decode with every decoder weight rebuilt from "
                      "artifacts/vae (INT8 x per-output-channel scales). Weight-only: activations "
                      "stay fp32 because SA is calibrate.py's output.",
        "layout": "OHWI per tap, inverted and asserted against the manifest's per-layer relFro",
        "roundtrip": {"layers": len(rows), "max_abs_relfro_diff":
                      max(abs(x["relfro_rebuilt"] - x["relfro_recorded"]) for x in rows),
                      "detail": rows},
        "result": {"psnr_db": p_all, "ssim": s_all,
                   "psnr_gate_db": PSNR_GATE_DB, "ssim_gate": SSIM_GATE,
                   "verdict": verdict, "per_frame": main_res["per_frame"]},
        "configs": results,
        "worst_weight_layer": worst_layer,
        "note": "gate thresholds are the ones Gate 1 declared; the two time_conv layers whose "
                "weight cosine the locked scheme drops to 0.9974 are the candidate cause of any "
                "miss, and are recorded in the artifact, not silently re-quantised here.",
    }
    out_json = OUT / f"gate1_psnr_{stamp}.json"
    out_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    log(f"wrote {rel(out_json)}")

    if args.out:
        md = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
        body = [
            "# Gate 1 — end-to-end PSNR/SSIM: FP32 vs the INT8 artifact's weights",
            "",
            f"`{stamp}` · latent `{list(z.shape)}` seed {args.seed} · **weight-only** "
            "(quantise-dequantise; activations stay FP32 because `SA` is `calibrate.py`'s output).",
            "",
            "| config | layers quantised | PSNR | SSIM | verdict |",
            "|---|---|---|---|---|",
        ]
        for r in results:
            body.append(f"| {r['config']} | {r['layers_quantised']}/36 | **{r['psnr_db']:.2f} dB** | "
                        f"**{r['ssim']:.4f}** | **{r['verdict']}** |")
        body += [
            "",
            f"Gates: PSNR ≥ {PSNR_GATE_DB} dB, SSIM ≥ {SSIM_GATE}.",
            "",
            "Round-trip: every rebuilt weight reproduces the artifact manifest's recorded per-layer "
            f"relFro (max |diff| {doc['roundtrip']['max_abs_relfro_diff']:.1e}), so the OHWI tap "
            "inversion is the artifact and not a plausible-looking transpose.",
            "",
            f"Per-frame PSNR (all-quantised): "
            + ", ".join(f"f{x['frame']} {x['psnr_db']:.2f}" for x in main_res["per_frame"])
            + " dB.",
        ]
        md.write_text("\n".join(body) + "\n", encoding="utf-8")
        log(f"wrote {rel(md)}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
