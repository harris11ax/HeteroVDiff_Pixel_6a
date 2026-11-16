#!/usr/bin/env python3
"""`convseq` driver — execute the VAE decoder's real INT8 artifact on the TPU.

`--decode 1` in the pipeline was a no-op stub: it printed the `T_V` = 10.640 s
hybrid reference and executed nothing. `phase1_quant/vae_int8.py` then wrote the
missing input (artifacts/vae/{weights_int8.bin,scales_f32.bin,bias_f32.bin,
vae_int8.json}). This script flattens that manifest into the line plan
`npu_block convseq` walks, pushes it with the artifact, runs it under the thermal
gate, and writes the artifact JSON + report.

Two questions this run answers on the device:

  1. Does this HAL accept `TENSOR_QUANT8_SYMM_PER_CHANNEL` for `CONV_2D`? It is
     rejected at `addOperation` for `FULLY_CONNECTED`, so the decode's ability to
     use the locked per-output-column scales at all is unmeasured.
  2. What does the decode actually cost when 592 `CONV_2D` dispatches execute the
     real artifact weights, against the 10.554 s per-op sum `T_V` is built from?

Nothing is modelled: an execute that did not happen is a count of 0, and a layer
that did not compile is named.

    python tools/npu/vae_seq.py --plan-only          # validate the chain, no device
    python tools/npu/vae_seq.py --build --run        # full 36-layer decode
    python tools/npu/vae_seq.py --run --layers 2 --pc 1   # prefix, force per-channel
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "probe"))
from probe_host import ADB, adb, bash_bin, pick_serial, read_zones, run, run_script  # noqa: E402

NPU_DIR = REPO / "tools" / "android" / "npublock"
BIN = NPU_DIR / "build" / "npu_block"
REMOTE_BIN = "/data/local/tmp/npu_block"
DEVICE_DIR = REPO / "docs" / "device"
VAE_DIR = REPO / "artifacts" / "vae"

PLAN_LOCAL = DEVICE_DIR / "vae_seq.txt"
R_SEQ = "/data/local/tmp/vae_seq.txt"
R_W = "/data/local/tmp/vae_int8.bin"
R_SC = "/data/local/tmp/vae_scales.f32"
R_BI = "/data/local/tmp/vae_bias.f32"
R_SA = "/data/local/tmp/vae_sa.f32"

ZONE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")
FATAL_STOP = ("OP_FAILED", "DEAD_OBJECT", "BAD_STATE")


def sha32(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def build_plan() -> tuple[list[dict], str]:
    """Flatten vae_int8.json + plan_npu.json into the device's line plan.

    The wiring code says how a layer's input frames are built from the previous
    layer's output: 0 = straight copy, 1 = temporal unshuffle (2C -> C over
    2t-1 frames), 2 = that plus a 2x nearest spatial upsample, 3 = synthetic
    latent (first layer only). It is *derived and asserted* here, so a plan/
    artifact disagreement is a hard error instead of a silently wrong execute.
    """
    art = json.loads((VAE_DIR / "vae_int8.json").read_text(encoding="utf-8"))
    plan = json.loads((REPO / "configs" / "vae" / "plan_npu.json").read_text(encoding="utf-8"))
    scale_bytes = art["files"]["scales_f32.bin"]["bytes"]
    bias_bytes = art["files"]["bias_f32.bin"]["bytes"]
    nscales = scale_bytes // 4
    assert nscales == bias_bytes // 4, "scale and bias tables must be the same length"

    layers, lines = [], []
    prev = None   # (frames, cout, h, w) of the previous layer's OUTPUT
    for i, (la, lp) in enumerate(zip(art["layers"], plan["layers"])):
        assert la["name"] == lp["name"], f"layer {i} name mismatch {la['name']} != {lp['name']}"
        cout, cin, h, w = la["taps"][0]["conv2d"]
        frames = lp["frames"]
        kh, kw = lp["taps"][0]["kernel"][1], lp["taps"][0]["kernel"][2]
        for t in la["taps"]:
            c2 = t["conv2d"]
            assert c2[0] == cout and c2[1] == cin and c2[2] == h and c2[3] == w, \
                f"{la['name']} tap {t['tap']} shape drift {c2}"
        for t in lp["taps"]:
            assert t["kernel"][1] == kh and t["kernel"][2] == kw, \
                f"{la['name']} tap kernel drift"
            assert t["conv2d"][0] == cout and t["conv2d"][1] == cin

        if prev is None:
            wiring = 3
        else:
            pf, pc, ph, pw = prev
            if (pf, pc, ph, pw) == (frames, cin, h, w):
                wiring = 0
            elif pc == 2 * cin and 2 * pf - 1 == frames and ph == h and pw == w:
                wiring = 1
            elif pc == 2 * cin and 2 * pf - 1 == frames and 2 * ph == h and 2 * pw == w:
                wiring = 2
            elif pc == cin and pf == frames and 2 * ph == h and 2 * pw == w:
                # spatial-only 2x nearest upsample: up_blocks.2's resample, after the
                # temporal expansion finished at up_blocks.1
                wiring = 5
            elif any((L["frames"], L["cout"], L["h"], L["w"]) == (frames, cin, h, w)
                     for L in layers):
                # a resnet shortcut: it consumes the enclosing resnet's entry buffer,
                # which an earlier layer produced at exactly this shape. The residual
                # add is not wired in this execution, so the conv runs on a zero
                # activation of that shape.
                wiring = 4
            else:
                raise SystemExit(
                    f"wiring: {la['name']} cannot consume the previous layer's "
                    f"[f={pf}, c={pc}, h={ph}, w={pw}] with its own "
                    f"[f={frames}, cin={cin}, h={h}, w={w}] and no earlier layer "
                    f"produced that input shape")

        sc_off_b = la["scales"]["offset"]
        assert sc_off_b % 4 == 0, f"{la['name']} scale offset {sc_off_b} not f4-aligned"
        # The manifest stores BYTE offsets (the device wants bytes for weights); the
        # device indexes the scale ARRAY, so divide the element size out exactly once
        # — the same trap that produced 72 spurious verifier failures in P3.8a.
        sc_off = sc_off_b // 4
        assert la["scales"]["count"] == cout, f"{la['name']} scale count {la['scales']['count']} != cout {cout}"
        assert sc_off + cout <= nscales, f"{la['name']} scale offset out of range"
        assert len(la["taps"]) == len(lp["taps"]) == lp["executes"] // frames, \
            f"{la['name']} tap count vs executes mismatch"

        lines.append(f"LAYER {i} {frames} {frames} "
                     f"{cout} {cin} {h} {w} {len(la['taps'])} {wiring} {sc_off} {la['name']}")
        for t, tp in zip(la["taps"], lp["taps"]):
            d = {"f": 0, "f-1": 1, "f-2": 2}[tp["input_frame"]]
            lines.append(f"TAP {t['tap']} {d} {t['offset']} {t['bytes']} {kh} {kw}")
        layers.append(dict(idx=i, name=la["name"], frames=frames, cout=cout, cin=cin, h=h, w=w,
                           taps=len(la["taps"]), wiring=wiring, sc_off=sc_off,
                           executes=lp["executes"], macs=lp["macs"],
                           plan_ms=lp.get("measured_ms_per_chunk")))
        prev = (frames, cout, h, w)
    return layers, "\n".join(lines) + "\n"


def parse(stdout: str) -> dict:
    rec = {"dev": [], "wfile": [], "seq": [], "probe": [], "note": [], "compile": [],
           "layer": [], "ts": [], "summary": [], "fatal": [], "done": 0}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        tag = parts[0]
        try:
            if tag == "TS":
                d = {}
                for kv in parts[2:]:          # parts[1] is the mode name, not a field
                    k, _, v = kv.partition("=")
                    d[k] = int(v) if k in ("l", "t", "f") else float(v)
                rec["ts"].append(d)
            elif tag == "CSLAYER":
                d = {}
                for kv in parts[1:]:          # parts[2] is the bare layer name
                    if "=" not in kv:
                        d["name"] = kv
                        continue
                    k, _, v = kv.partition("=")
                    d[k] = int(v) if k in ("layer", "frames", "taps", "executes", "macs", "pc") else float(v)
                rec["layer"].append(d)
            elif tag in ("SUMMARY",):
                d = {}
                for kv in parts[1:]:
                    k, _, v = kv.partition("=")
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
                rec["summary"].append(d)
            elif tag == "CSCOMPILE":
                d = {}
                for kv in parts[2:]:
                    k, _, v = kv.partition("=")
                    d[k] = v
                rec["compile"].append(d)
            elif tag in ("SEQ", "PROBE", "NOTE"):
                rec[tag.lower()].append(line)
            elif tag == "WFILE":
                rec["wfile"].append(dict(kv.split("=", 1) for kv in parts[1:] if "=" in kv))
            elif tag == "DEV":
                rec["dev"].append(dict(kv.split("=", 1) for kv in parts[1:] if "=" in kv))
            elif tag == "FATAL":
                rec["fatal"].append(line)
            elif tag == "DONE":
                rec["done"] += 1
        except (IndexError, ValueError):
            continue
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--layers", type=int, default=0, help="0 = all 36")
    ap.add_argument("--pc", type=int, default=-1, choices=[-1, 0, 1],
                    help="-1 = probe SYMM_PER_CHANNEL on CONV_2D first")
    ap.add_argument("--cooldown", action="store_true",
                    help="refuse to run unless max(TPU, skin, VIRTUAL-SKIN) <= battery + 3")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--out", default="vae_seq")
    args = ap.parse_args()

    layers, text = build_plan()
    PLAN_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    PLAN_LOCAL.write_text(text, encoding="utf-8")
    print(f"plan: {len(layers)} layers, {sum(L['taps'] for L in layers)} taps, "
          f"{sum(L['executes'] for L in layers)} executes -> {PLAN_LOCAL}")
    for L in layers:
        print(f"  [{L['idx']:2d}] {L['name'][:46]:46s} f={L['frames']} {L['cin']}x{L['h']}x{L['w']}"
              f"->{L['cout']} taps={L['taps']} wiring={L['wiring']} ex={L['executes']:3d}"
              f" plan={L['plan_ms']}")
    if args.plan_only:
        return 0

    if args.build:
        rc, txt = run_script(NPU_DIR / "build.sh", timeout=900)
        print(txt[-1500:])
        if rc != 0:
            print("build failed", file=sys.stderr)
            return rc
    if not BIN.exists():
        print("npu_block not built — pass --build", file=sys.stderr)
        return 2
    if not args.run:
        return 0

    serial = pick_serial()
    if not serial:
        print("no device", file=sys.stderr)
        return 2

    zones0 = read_zones(serial)
    peak = max((v for k, v in zones0.items() if k in ZONE_KEYS), default=None)
    batt = zones0.get("battery")
    gate = (batt + 3.0) if batt is not None else None
    verdict = ("no battery zone" if gate is None else
               ("PASS" if peak is not None and peak <= gate else "OVER"))
    print(f"thermal gate: peak={peak} C  battery+3={gate} C -> {verdict}")
    if args.cooldown and verdict != "PASS":
        print("TOO WARM — refusing to time a hot run", file=sys.stderr)
        return 3

    for src, dst in ((VAE_DIR / "weights_int8.bin", R_W),
                     (VAE_DIR / "scales_f32.bin", R_SC),
                     (VAE_DIR / "bias_f32.bin", R_BI),
                     (PLAN_LOCAL, R_SEQ),
                     (BIN, REMOTE_BIN)):
        rc, txt = run([str(ADB), "-s", serial, "push", str(src), dst], timeout=300)
        if rc != 0:
            print(f"push failed for {src}: {txt}", file=sys.stderr)
            return 2
        print(f"pushed {src.name} -> {dst}")
    # The measured per-layer activation scale (phase1_quant/vae_sa.py). Optional:
    # without it the device falls back to 1/127 and says `sa=nominal`.
    sa_local = VAE_DIR / "act_scales_f32.bin"
    sa_sha = None
    if sa_local.exists():
        rc, txt = run([str(ADB), "-s", serial, "push", str(sa_local), R_SA], timeout=120)
        if rc != 0:
            print(f"push failed for {sa_local}: {txt}", file=sys.stderr)
            return 2
        sa_sha = sha32(sa_local)
        print(f"pushed {sa_local.name} ({sa_local.stat().st_size} B, sha {sa_sha}) -> {R_SA}")
    else:
        print("WARNING: no act_scales_f32.bin — the device will run at sa=nominal")
    adb(serial, f"shell chmod 755 {REMOTE_BIN}")

    adb(serial, "logcat -c")
    cmd = f"shell {REMOTE_BIN} convseq {args.layers} {args.pc}"
    print(f"running: {cmd}")
    rc, out = adb(serial, cmd, timeout=args.timeout)
    print(out)
    zones1 = read_zones(serial)
    _, log = adb(serial, "logcat -d -s NeuralNetworks:* NnapiHal:* nnapi:* -t 160", timeout=60)

    rec = parse(out)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    doc = {"serial": serial, "timestamp": ts, "cmd": cmd, "rc": rc,
           "bin_sha": sha32(BIN), "plan_sha": hashlib.sha256(text.encode()).hexdigest()[:16],
           "weights_sha": sha32(VAE_DIR / "weights_int8.bin"),
           "act_scales_sha": sa_sha,
           "layers_planned": layers, "layers": args.layers, "pc": args.pc,
           "thermal": {"before": zones0, "after": zones1, "gate": gate, "verdict": verdict},
           "parsed": rec, "logcat_tail": log[-4000:]}
    js = DEVICE_DIR / f"{args.out}_{ts}.json"
    js.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    s = (rec["summary"] or [{}])[0]
    md = [f"# VAE decode execution (`convseq`) — {ts}", "",
          f"serial `{serial}` · binary `{doc['bin_sha']}` · weights `{doc['weights_sha']}` · "
          f"act-scales `{sa_sha or 'none (nominal)'}` · "
          f"layers={args.layers or 'all'} pc={args.pc} rc={rc}", "",
          "## Thermal", "",
          "- before: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(zones0.items()) if k in ZONE_KEYS),
          "- after:  " + " ".join(f"{k}={v:.1f}" for k, v in sorted(zones1.items()) if k in ZONE_KEYS),
          f"- gate max(TPU,skin,VIRTUAL-SKIN) <= battery+3 = {gate} -> **{verdict}**", ""]
    if rec["probe"]:
        md += ["## The fork: per-channel scales on `CONV_2D`", "", "```"] + rec["probe"] + ["```", ""]
    if s:
        md += ["## Result", "", "| quantity | value |", "|---|---|",
               f"| layers executed | {s.get('layers')} |",
               f"| `CONV_2D` executes | **{s.get('executes')}** |",
               f"| compile | {s.get('compile_s')} s |",
               f"| wall (whole run) | **{s.get('wall_s')} s** |",
               f"| MACs | {s.get('macs')} |",
               f"| effective | {s.get('gops')} GOPS |",
               f"| per-channel used | {s.get('pc')} (fallbacks {s.get('pc_fallbacks')}) |",
               f"| activation scale | **{s.get('sa')}** (table {sa_sha or '—'}) |",
               f"| bias int32 clamps | {s.get('bias_clamps')} |", ""]
    if rec["layer"]:
        md += ["## Per layer", "",
               "| layer | frames | taps | executes | ms | GOPS | plan ms | SA | epi ms | clamp |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for L in rec["layer"]:
            pl = next((x["plan_ms"] for x in layers if x["name"] == L["name"]), None)
            md.append(f"| `{L['name']}` | {L['frames']} | {L['taps']} | {L['executes']} | "
                      f"{L['ms']:.1f} | {L['gops']:.1f} | {pl} | {L.get('sa', float('nan')):.5f} | "
                      f"{L.get('epi_ms', float('nan')):.1f} | {L.get('clamp', float('nan')) * 100:.2f} % |")
        md.append("")
        tot_ms = sum(L["ms"] for L in rec["layer"])
        tot_epi = sum(L.get("epi_ms") or 0.0 for L in rec["layer"])
        if tot_ms > 0:
            md += [f"Epilogue share of the layer walk: **{tot_epi / tot_ms * 100:.1f} %** "
                   f"({tot_epi / 1000:.2f} s of {tot_ms / 1000:.2f} s). "
                   f"Worst saturation {max((L.get('clamp') or 0.0) for L in rec['layer']) * 100:.2f} %.", ""]
    if rec["fatal"]:
        md += ["## Failures", "", "```"] + rec["fatal"] + ["```", ""]
    md += ["<details><summary>logcat (NNAPI)</summary>", "", "```", log[-2500:], "```", "",
           "</details>", ""]
    (DEVICE_DIR / f"{args.out}_{ts}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {js}")
    print(f"wrote {DEVICE_DIR / (args.out + '_' + ts + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
