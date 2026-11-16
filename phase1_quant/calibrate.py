#!/usr/bin/env python3
"""Deliverable 4 — activation calibration: `SA = max|x| / 127` per GEMM / per layer.

`SA` is the one number the whole INT8 story is missing. Everything the device
does leans on it: the GPU's fp16 hidden-state narrowing ("per-GEMM calibrated
input scale"), the NPU's INT8 activation operand, the VAE artifact's INT32 bias
(`SA*SW` units — which is *why* `artifacts/vae/bias_f32.bin` ships unquantised),
and any comparison between the 40.22 dB weight-only Gate 1 number and what the
INT8 activation path actually costs. Until now `npu_block` used the declared
nominal `1/127`, which is a placeholder, not a measurement.

Two targets, both FP32 on the host in `.venv-ref`, no device:

  vae  — one seeded decode through `AutoencoderKLWan` with a pre-hook on every
         one of the 36 decoder conv layers the INT8 artifact carries, so the
         scale is measured on the activation that layer's taps consumed.
  dit  — the B=180 trajectory as it actually is: 6 timesteps (the locked DCM
         schedule) x 30 blocks, a forward of `WanTransformer3DModel` at
         Preset A (M = 1530 tokens), with a pre-hook on the block's 10 base
         GEMMs => 180 x 10 = 1800 scales, keyed by (step, block, GEMM).

Honesty rules that are enforced in code, not asserted in prose:

  * the loaded state dict is *proven*: every stock parameter must be filled and
    the count of dropped tensors (the `lora_A`/`lora_B` factors, device-side
    FP16 experts that are not part of the INT8 base) is reported, per role;
  * hooks record `max|x|` over the tensor the GEMM really consumed — the value
    written is `max|x|/127`, and nothing is modelled or interpolated;
  * a layer whose hook never fired is an error, not a row of zeros;
  * the text conditioning source is recorded in the artifact. `--text real`
    runs umT5-XXL; if that cannot load, the run says so and uses seeded
    synthetic conditioning with `"text_source"` naming the failure — it never
    silently substitutes.

    .venv-ref/Scripts/python.exe phase1_quant/calibrate.py --target vae
    .venv-ref/Scripts/python.exe phase1_quant/calibrate.py --target dit --max-steps 1
    .venv-ref/Scripts/python.exe phase1_quant/calibrate.py --target both --out docs/device/RESULTS_calibrate.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "weights_raw" / "DCM_WAN"
VAE_ART = REPO / "artifacts" / "vae"
CAL = REPO / "artifacts" / "calib"
OUT = REPO / "docs" / "device"
LOCKS = REPO / "configs" / "design_locks.json"

LATENT = (1, 16, 3, 34, 60)                 # Preset A: 9 frames x 272 x 480
GEMMS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
         "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
         "ffn.net.0.proj", "ffn.net.2")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def quant_stats(x: np.ndarray, sa: float) -> dict:
    """What INT8 at this SA costs the tensor the GEMM actually consumed."""
    q = np.clip(np.rint(x / sa), -127.0, 127.0)
    dq = (q * sa).astype(np.float32, copy=False)
    n = float(np.linalg.norm(x))
    err = float(np.linalg.norm(dq - x) / n) if n > 0 else 0.0
    return {"rel_fro_qdq": err,
            "saturated_frac": float(np.count_nonzero(np.abs(q) >= 127.0) / q.size),
            "zero_frac": float(np.count_nonzero(q == 0.0) / q.size)}


# --------------------------------------------------------------------- VAE
def calib_vae(seed: int, frames: int) -> dict:
    import torch
    from diffusers import AutoencoderKLWan

    art = json.loads((VAE_ART / "vae_int8.json").read_text(encoding="utf-8"))
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(str(MODEL / "vae"), torch_dtype=torch.float32).eval()
    log(f"vae: reference loaded in {time.time() - t0:.1f} s")

    rec: dict[str, dict] = {}
    handles = []

    def make(name: str, kind: str):
        def hook(_mod, inp):
            x = inp[0].detach()
            a = x.numpy().astype(np.float32, copy=False)
            m = float(np.abs(a).max())
            # `decode` walks the latent iteratively with cached conv state, so one
            # layer fires once per frame chunk: keep the max over every call, not
            # the last one — the SA must survive the worst chunk.
            r = rec.setdefault(name, {"kind": kind, "shape": list(x.shape),
                                      "max_abs": 0.0, "calls": 0,
                                      "per_call_max": [], "per_frame_max": [],
                                      "mean_abs_acc": [], "n": 0})
            r["calls"] += 1
            r["max_abs"] = max(r["max_abs"], m)
            r["per_call_max"].append(m)
            r["per_frame_max"] = [max(o, n_) for o, n_ in
                                  zip(r["per_frame_max"] or [0.0] * x.shape[2],
                                      [float(x[:, :, f].abs().max()) for f in range(x.shape[2])])]
            r["shape"] = list(x.shape)
            r["mean_abs_acc"].append(float(np.abs(a).mean()))
            r["n"] += int(a.size)
        return hook

    for L in art["layers"]:
        mod = vae.get_submodule(L["name"])
        handles.append(mod.register_forward_pre_hook(make(L["name"], type(mod).__name__)))
    log(f"vae: {len(handles)} pre-hooks on the artifact's layers")

    g = torch.Generator().manual_seed(seed)
    z = torch.randn(1, 16, frames, LATENT[3], LATENT[4], generator=g)
    t0 = time.time()
    with torch.no_grad():
        out = vae.decode(z).sample
    log(f"vae: decode {tuple(z.shape)} -> {tuple(out.shape)} in {time.time() - t0:.1f} s")

    missing = [L["name"] for L in art["layers"] if L["name"] not in rec]
    if missing:
        raise SystemExit(f"vae: {len(missing)} layer(s) never fired a hook: {missing[:4]}")
    for h in handles:
        h.remove()

    rows = []
    for L in art["layers"]:
        r = rec[L["name"]]
        rows.append({"layer": L["name"], "kind": r["kind"], "in_shape": r["shape"],
                     "calls": r["calls"], "max_abs": r["max_abs"],
                     "mean_abs": float(np.mean(r["mean_abs_acc"])),
                     "sa": r["max_abs"] / 127.0 if r["max_abs"] > 0 else None,
                     "sa_per_frame": [m / 127.0 for m in r["per_frame_max"]],
                     "per_frame_max": r["per_frame_max"],
                     "per_call_max": r["per_call_max"],
                     "max_abs_out": float(out.abs().max()),
                     "weight_relfro": float(L["quant"]["rel_fro"]),
                     "weight_cosine": float(L["quant"]["cosine"])})
    return {"target": "vae", "seed": seed, "latent": list(z.shape),
            "layers": rows, "built": time.strftime("%Y%m%d_%H%M%S"),
            "note": "SA = max|x|/127 over the tensor the layer's conv taps consumed, "
                    "FP32 forward, real checkpoint weights. The artifact's FP32 bias can "
                    "now become INT32 in SA*SW units; nothing here quantises the decode."}


# --------------------------------------------------------------------- DiT
def map_dit_state(dit_dir: Path) -> tuple[dict, dict]:
    """Checkpoint -> stock WanTransformer3DModel names, with a role census."""
    from safetensors.torch import load_file

    sd, census = {}, Counter()
    files = sorted(dit_dir.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors under {dit_dir}")
    for f in files:
        for k, v in load_file(str(f)).items():
            k = k.replace("diffusion_model.", "", 1)
            if ".lora_A." in k or ".lora_B." in k:
                census["expert_lora_dropped"] += 1
                continue
            if ".base_layer." in k:
                sd[k.replace(".base_layer.", ".")] = v
                census["base_layer_mapped"] += 1
                continue
            sd[k] = v
            census["direct"] += 1
    return sd, dict(census)


def calib_dit(args) -> dict:
    import torch
    from diffusers import WanTransformer3DModel

    torch.set_grad_enabled(False)
    locks = json.loads(LOCKS.read_text(encoding="utf-8"))
    steps = [int(t) for t in locks["model"]["timesteps"]][: args.max_steps or None]
    nblocks = locks["model"]["L_blocks"]
    log(f"dit: locked schedule {steps} over {nblocks} blocks")

    cfg = WanTransformer3DModel.load_config(str(MODEL / "transformer"))

    # -- text conditioning FIRST: umT5-XXL is 22.7 GB on disk (11 GB in bf16) and must
    # not be co-resident with the transformer, so it is encoded, freed, and only then
    # is the DiT built. Both the source and any failure land in the artifact.
    enc, have, text_src = None, False, ""
    if args.text == "real":
        try:
            from transformers import AutoTokenizer, UMT5EncoderModel
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(str(MODEL / "tokenizer"))
            enc = UMT5EncoderModel.from_pretrained(str(MODEL / "text_encoder"),
                                                  torch_dtype=torch.bfloat16,
                                                  low_cpu_mem_usage=True).eval()
            log(f"dit: umT5-XXL loaded bf16 in {time.time() - t0:.1f} s")
            ids = tok([args.prompt], padding="max_length", max_length=512,
                      truncation=True, return_tensors="pt").input_ids
            t0 = time.time()
            with torch.no_grad():
                eh = enc(ids).last_hidden_state.to(torch.float32)
            log(f"dit: umT5-XXL text embedding {tuple(eh.shape)} in {time.time() - t0:.1f} s")
            text_src = (f"umT5-XXL bf16 encode of prompt {args.prompt!r}, "
                        f"{int(ids.shape[1])} tokens; max|e| {float(eh.abs().max()):.3f}")
            have = True
            del enc, tok, ids
            enc = None
            import gc
            gc.collect()
        except Exception as e:                      # noqa: BLE001 - recorded, not hidden
            text_src = f"umT5 load/encode FAILED: {type(e).__name__}: {e}"
            log(f"dit: WARNING {text_src}")
            enc = None
    if not have:
        g = torch.Generator().manual_seed(args.seed + 7)
        eh = torch.randn(1, 512, cfg["text_dim"], generator=g) * 0.5
        text_src = (("synthetic seeded N(0, 0.5**2) [1,512,4096] after " + text_src + "; ")
                    if text_src else
                    "synthetic seeded N(0, 0.5**2) [1,512,4096] — NOT the umT5 output; ") + \
                   "run with --text real for the true conditioning"

    t0 = time.time()
    model = WanTransformer3DModel.from_config(cfg).eval().to(torch.float32)
    log(f"dit: instantiated from config in {time.time() - t0:.1f} s")

    sd, census = map_dit_state(MODEL / "transformer")
    t0 = time.time()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd
    log(f"dit: state dict {census} | load {time.time() - t0:.1f} s | "
        f"missing {len(missing)} unexpected {len(unexpected)}")
    if missing:
        raise SystemExit(f"dit: {len(missing)} stock parameter(s) never filled: {missing[:6]}")
    unexpected = [u for u in unexpected if not u.endswith("num_batches_tracked")]
    if unexpected:
        log(f"dit: {len(unexpected)} unexpected tensor(s), first {unexpected[:4]}")

    if args.dry_map:
        return {"target": "dit", "dry_map": True, "census": census,
                "missing": 0, "unexpected": list(unexpected)[:20]}

    # -- noise level per locked timestep: rectified flow, sigma = t/1000.
    # The scheduler's own set_timesteps(n) is *not* the DCM distilled schedule (it returns
    # uniformly spaced timesteps), but its sigma convention is the identity sigma = t/1000
    # (measured: 999->1.0000, 937->0.9376, 857->0.8573, 750->0.7504, 600->0.6007, 376->0.3764),
    # so the locked timesteps get the matching noise level from that same convention.
    from diffusers import UniPCMultistepScheduler
    sched = UniPCMultistepScheduler.from_pretrained(str(MODEL / "scheduler"))
    sched.set_timesteps(len(steps))
    sched_ts = [int(round(float(t))) for t in sched.timesteps]
    sched_sig = [float(s) for s in sched.sigmas][: len(steps)]
    identity_err = max(abs(s - min(1.0, t / 1000.0)) for s, t in zip(sched_sig, sched_ts))
    sig = [min(1.0, t / 1000.0) for t in steps]
    log(f"dit: locked timesteps {steps} -> sigmas {['%.4f' % s for s in sig]}; "
        f"scheduler cross-check {sched_ts} / {['%.4f' % s for s in sched_sig]} "
        f"(identity max err {identity_err:.4f})")

    rec: dict[tuple[int, int, str], dict] = {}
    hs_rec: dict[tuple[int, int], float] = {}
    handles = []

    for bi, blk in enumerate(model.blocks):
        def mk_gemm(b, name):
            def hook(_m, inp):
                a = inp[0].detach().numpy()
                m = float(np.abs(a).max())
                rec[(steps[cursor[0]], b, name)] = {
                    "shape": list(a.shape), "max_abs": m, "mean_abs": float(np.abs(a).mean()),
                    "sa": m / 127.0 if m > 0 else None}
            return hook

        def mk_hs(b):
            def hook(_m, inp):
                a = inp[0].detach().numpy()
                hs_rec[(steps[cursor[0]], b)] = float(np.abs(a).max())
            return hook

        handles.append(blk.register_forward_pre_hook(mk_hs(bi)))
        for gname in GEMMS:
            mod = blk.get_submodule(gname)
            handles.append(mod.register_forward_pre_hook(mk_gemm(bi, gname)))
    cursor = [0]
    log(f"dit: {len(handles)} hooks ({len(GEMMS)} GEMMs x {len(model.blocks)} blocks + hidden state)")

    g = torch.Generator().manual_seed(args.seed)
    base = torch.randn(1, 16, LATENT[2], LATENT[3], LATENT[4], generator=g)   # z_0 stand-in

    per_step = []
    for si, (t, sg) in enumerate(zip(steps, sig)):
        cursor[0] = si
        g2 = torch.Generator().manual_seed(args.seed * 1000 + si)
        noise = torch.randn(base.shape, generator=g2)
        x = (1.0 - sg) * base + sg * noise
        timestep = torch.tensor([float(t)])
        t0 = time.time()
        with torch.no_grad():
            out = model(hidden_states=x, timestep=timestep, encoder_hidden_states=eh,
                        return_dict=False)[0]
        log(f"dit: step {si + 1}/{len(steps)} t={t} sigma={sg:.4f} -> "
            f"{tuple(out.shape)} max|y| {float(out.abs().max()):.3f} in {time.time() - t0:.1f} s")
        per_step.append({"step": si, "timestep": t, "sigma": sg,
                         "out_max_abs": float(out.abs().max()),
                         "out_mean_abs": float(out.abs().mean()),
                         "input_max_abs": float(x.abs().max()),
                         "input_shape": list(x.shape)})

    for h in handles:
        h.remove()

    want = len(steps) * len(model.blocks) * len(GEMMS)
    if len(rec) != want:
        raise SystemExit(f"dit: {len(rec)} GEMM hooks fired, expected {want} "
                         f"({len(steps)} steps x {len(model.blocks)} blocks x {len(GEMMS)} GEMMs)")

    rows = []
    for (t, b, name), r in sorted(rec.items(), key=lambda kv: (kv[0][0] == 0, kv[0][0],
                                                              kv[0][1], GEMMS.index(kv[0][2]))):
        rows.append({"timestep": t, "block": b, "gemm": name, "in_shape": r["shape"],
                     "max_abs": r["max_abs"], "mean_abs": r["mean_abs"], "sa": r["sa"]})
    by_gemm = defaultdict(list)
    for r in rows:
        by_gemm[r["gemm"]].append(r["sa"])
    summary = {g: {"sa_min": min(v), "sa_max": max(v), "sa_median": float(np.median(v)),
                   "n": len(v), "spread": max(v) / min(v)} for g, v in by_gemm.items()}
    hs = [{"timestep": t, "block": b, "max_abs": m} for (t, b), m in sorted(hs_rec.items())]
    if len(hs) != len(steps) * len(model.blocks):
        raise SystemExit(f"dit: {len(hs)} hidden-state hooks, expected "
                         f"{len(steps) * len(model.blocks)}")

    return {"target": "dit", "seed": args.seed, "schedule": steps, "sigmas": sig,
            "scheduler_timesteps": sched_ts, "scheduler_sigmas": sched_sig,
            "sigma_identity_max_err": identity_err,
            "tokens": int(1530), "latent": list(base.shape),
            "text_source": text_src, "state_dict_census": census,
            "dropped_unexpected": unexpected[:20],
            "per_step": per_step, "blocks": len(model.blocks), "gemms": list(GEMMS),
            "rows": rows, "by_gemm": summary, "hidden_state": hs,
            "built": time.strftime("%Y%m%d_%H%M%S"),
            "note": "SA = max|x|/127 of the tensor each base GEMM consumed, FP32 forward of the "
                    "FP32 base weights (no expert LoRA: the INT8 base is base-only). Latents are "
                    "the step's sigma mix of a seeded z_0 stand-in and seeded noise — a sample of "
                    "the trajectory's activation regime, not a decoded video."}


# --------------------------------------------------------------------- report
def write_md(vae: dict | None, dit: dict | None, path: Path) -> None:
    b = ["# Calibration — `SA = max|x| / 127`, the activation scale (deliverable 4)", "",
         f"built `{time.strftime('%Y%m%d_%H%M%S')}` on `.venv-ref` (torch 2.15.0+cpu), host only, "
         "no device involved.", ""]
    if vae:
        sas = [r["sa"] for r in vae["layers"]]
        b += ["## VAE decoder — 36 layers", "",
              f"latent `{vae['latent']}` seed {vae['seed']}; every layer's scale is measured on the "
              "activation its own taps consumed.", "",
              f"`SA` range **{min(sas):.6f} … {max(sas):.6f}** (spread {max(sas) / min(sas):.2f}x) "
              f"against the placeholder `1/127 = {1/127:.6f}`.", "",
              "| layer | in shape | max&#124;x&#124; | SA | vs 1/127 |", "|---|---|---|---|---|"]
        for r in vae["layers"]:
            b.append(f"| `{r['layer'].replace('decoder.', '')}` | {r['in_shape']} | "
                     f"{r['max_abs']:.3f} | {r['sa']:.6f} | {r['sa'] * 127:.2f}x |")
        b.append("")
    if dit:
        b += ["## DiT block GEMMs — 6 steps x 30 blocks x 10 GEMMs", "",
              f"text conditioning: {dit['text_source']}", "",
              f"schedule {dit['schedule']} (locked), flow sigmas "
              f"{['%.3f' % s for s in dit['sigmas']]}, M = {dit['tokens']} tokens.", "",
              "| GEMM | SA min | SA median | SA max | spread |", "|---|---|---|---|---|"]
        for gname, s in dit["by_gemm"].items():
            b.append(f"| `{gname}` | {s['sa_min']:.6f} | {s['sa_median']:.6f} | "
                     f"{s['sa_max']:.6f} | {s['spread']:.2f}x |")
        hs = [h["max_abs"] for h in dit["hidden_state"]]
        b += ["", f"Residual stream max&#124;x&#124; {min(hs):.2f} … {max(hs):.2f} across all "
                  f"{len(hs)} (step, block) entries.", ""]
    path.write_text("\n".join(b) + "\n", encoding="utf-8")
    log(f"wrote {rel(path)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("vae", "dit", "both"), default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=LATENT[2])
    ap.add_argument("--max-steps", type=int, default=0, help="limit the DiT schedule (0 = all 6)")
    ap.add_argument("--text", choices=("real", "synthetic"), default="synthetic")
    ap.add_argument("--prompt", default="a red fox walking through fresh snow at golden hour")
    ap.add_argument("--dry-map", action="store_true", help="DiT: check the state-dict mapping only")
    ap.add_argument("--out", default="docs/device/RESULTS_calibrate.md")
    args = ap.parse_args()

    CAL.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    vae = dit = None
    if args.target in ("vae", "both"):
        vae = calib_vae(args.seed, args.frames)
        p = CAL / f"act_scales_vae_{stamp}.json"
        p.write_text(json.dumps(vae, indent=1), encoding="utf-8")
        log(f"wrote {rel(p)}")
    if args.target in ("dit", "both"):
        dit = calib_dit(args)
        p = CAL / f"act_scales_dit_{stamp}.json"
        p.write_text(json.dumps(dit, indent=1), encoding="utf-8")
        log(f"wrote {rel(p)}")
    if not args.dry_map:
        write_md(vae, dit, (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
