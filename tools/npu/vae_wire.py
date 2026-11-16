#!/usr/bin/env python3
"""`convwire` driver — execute the VAE decode WITH the P3.11 wiring on the TPU.

P3.10 wired the measured per-layer activation scale (SA) into `convseq` and got
592/592 executes for 8,822.6 ms, but the walk still had no norm, no SiLU, no
residual add — so it executed without decoding. P3.11 then measured what those
ops *are* (a per-pixel `WanRMS_norm` + SiLU between the two convs of every
resnet, 29 norms / 29 SiLUs / 14 adds / 2 temporal shuffles / 3 2-D upsamples /
1 mid-block attention) and wrote the 128-op plan to
`artifacts/vae/vae_wiring.json`.

This tool turns that plan into a **linear program** the device walks
(`docs/device/vae_wire.txt`), in EXECUTION order, and runs it as
`npu_block convwire`. Three things it fixes relative to `convseq`:

  1. **Execution order, not storage order.** The artifact indexes each resnet as
     `conv1, conv2, conv_shortcut`; `WanResidualBlock.forward` applies the
     shortcut FIRST. The program is emitted by walking `vae_wiring.json`'s op
     list, so every conv carries its own `artifact_index` for the SA / scale /
     bias tables instead of relying on position.
  2. **Both scales per layer.** `act_scales2_f32.bin` = SA_in ++ SA_out, so the
     conv's output operand carries the layer's measured OUTPUT scale (it used to
     be quantised at SA_in and rescaled to the consumer's, which throws away
     resolution whenever SA_out < SA_in).
  3. **The float ops.** norm (L2 over channels x sqrt(C) x gamma), SiLU, and the
     residual add at the measured `post_add_sa` are real passes over the
     activation. Only the convs go to NNAPI; everything else is a CPU float pass,
     which is what the plan says ("only the conv ops are INT8/NNAPI").

The program is *derived and asserted* here — shapes as it walks, op counts
against the plan — so a plan/artifact disagreement is a hard error before the
device is touched.

    python tools/npu/vae_wire.py --plan-only         # derive + assert, no device
    python tools/npu/vae_wire.py --build --run       # build binary, full 36-layer walk
    python tools/npu/vae_wire.py --run --convs 3     # 3-conv prefix smoke
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
from probe_host import ADB, adb, pick_serial, read_zones, run, run_script  # noqa: E402

NPU_DIR = REPO / "tools" / "android" / "npublock"
BIN = NPU_DIR / "build" / "npu_block"
REMOTE_BIN = "/data/local/tmp/npu_block"
DEVICE_DIR = REPO / "docs" / "device"
VAE_DIR = REPO / "artifacts" / "vae"

PLAN_LOCAL = DEVICE_DIR / "vae_wire.txt"
R_W = "/data/local/tmp/vae_int8.bin"
R_SC = "/data/local/tmp/vae_scales.f32"
R_BI = "/data/local/tmp/vae_bias.f32"
R_WIRE = "/data/local/tmp/vae_wire.txt"
R_SA2 = "/data/local/tmp/vae_sa2.f32"
R_NM = "/data/local/tmp/vae_norms.f32"

ZONE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")

TAPS_WANT = dict(conv=36, norm=29, silu=29, add=14, save=14,
                 temporal_shuffle=2, upsample2d=3, attn=1)


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def f32(path: Path) -> list[float]:
    raw = path.read_bytes()
    assert len(raw) % 4 == 0, f"{path.name} not f32-aligned ({len(raw)} B)"
    return list(__import__("array").array("f", raw))


def build_wire() -> tuple[list[dict], list[tuple], str, dict]:
    """Walk the P3.11 plan and emit the device's linear program."""
    art = json.loads((VAE_DIR / "vae_int8.json").read_text(encoding="utf-8"))
    plan = json.loads((REPO / "configs" / "vae" / "plan_npu.json").read_text(encoding="utf-8"))
    wire = json.loads((VAE_DIR / "vae_wiring.json").read_text(encoding="utf-8"))
    ops = wire["ops"]
    nsa = len(f32(VAE_DIR / "act_scales2_f32.bin"))
    assert nsa == 2 * len(art["layers"]), f"act_scales2 has {nsa} floats, want 2x{len(art['layers'])}"
    SAIN = f32(VAE_DIR / "act_scales2_f32.bin")[:len(art["layers"])]
    SAOUT = f32(VAE_DIR / "act_scales2_f32.bin")[len(art["layers"]):]
    n_gamma = len(f32(VAE_DIR / "norms_f32.bin"))
    assert n_gamma == int(wire["n_gamma_floats"]), \
        f"norms_f32.bin has {n_gamma} floats, wiring.json says {wire['n_gamma_floats']}"

    # ---- the artifact, indexed by NAME (its list order is a storage order) ----
    by_name: dict[str, dict] = {}
    for i, la in enumerate(art["layers"]):
        cout, cin, h, w = la["taps"][0]["conv2d"]
        for t in la["taps"]:
            assert t["conv2d"] == [cout, cin, h, w], f"{la['name']} tap shape drift"
        sc_off_b = la["scales"]["offset"]
        assert sc_off_b % 4 == 0, f"{la['name']} scale offset not f4"
        assert la["scales"]["count"] == cout
        by_name[la["name"]] = dict(
            artifact_index=i, cout=cout, cin=cin, h=h, w=w, taps=la["taps"],
            sc_off=sc_off_b // 4, sa_in=SAIN[i], sa_out=SAOUT[i])
    plan_by_name = {lp["name"]: lp for lp in plan["layers"]}

    counts: dict[str, int] = {}
    layers: list[dict] = []          # execution order, index == the CONV op's `li`
    prog: list[tuple] = []
    lines: list[str] = []
    logs: list[str] = []

    # symbolic activation: (frames, ch, h, w) and its scale; mirrors the device
    act = None                        # None = the decode's synthetic latent
    act_sa = None
    slot: dict[int, tuple | None] = {0: None, 1: None}

    def emit(op: str, *rest) -> None:
        prog.append((op,) + rest)

    def conv_op(op_json: dict) -> None:
        nonlocal act, act_sa
        name = op_json["name"]
        a = by_name[name]
        lp = plan_by_name[name]
        frames = lp["frames"]
        fin = frames
        need = (fin, a["cin"], a["h"], a["w"])
        # `act` is the DEVICE BUFFER's shape — the shuffle/upsample the plan lists
        # are folded into the consumer conv's input gather, so the buffer itself is
        # still the pre-transform tensor here. The wiring code below is exactly
        # that fold, so the tests are against the buffer shape.
        h2, w2 = a["h"] // 2, a["w"] // 2
        if act is None:
            wiring = 3
        elif act == need:
            wiring = 0
        elif act[0] * 2 - 1 == fin and act[1] == 2 * a["cin"] and act[2] == a["h"] and act[3] == a["w"]:
            wiring = 1                       # temporal unshuffle 2C -> C, 2t-1 frames
        elif act[0] * 2 - 1 == fin and act[1] == 2 * a["cin"] and act[2] == h2 and act[3] == w2:
            wiring = 2                       # ... plus 2x nearest spatial upsample
        elif act[0] == fin and act[1] == a["cin"] and act[2] == h2 and act[3] == w2:
            wiring = 5                       # 2x nearest upsample only
        else:
            raise SystemExit(f"{name}: cannot consume activation {act} -> needs {need}")
        kh, kw = lp["taps"][0]["kernel"][1], lp["taps"][0]["kernel"][2]
        assert len(a["taps"]) == len(lp["taps"]), f"{name} tap count"
        li = len(layers)
        # A residual branch reads the resnet's ENTRY, which is the activation as
        # it stands before this conv — capture it before the conv consumes it.
        shortcut = name.endswith("conv_shortcut")
        if shortcut:
            save_slot(0)
        lines.append(f"LAYER {li} {frames} {fin} {a['cout']} {a['cin']} {a['h']} {a['w']} "
                     f"{len(a['taps'])} {wiring} {a['sc_off']} {a['artifact_index']} {name}")
        for t, tp in zip(a["taps"], lp["taps"]):
            d = {"f": 0, "f-1": 1, "f-2": 2}[tp["input_frame"]]
            assert tp["kernel"][1] == kh and tp["kernel"][2] == kw, f"{name} kernel drift"
            lines.append(f"TAP {t['tap']} {d} {t['offset']} {t['bytes']} {kh} {kw}")
        layers.append(dict(idx=li, name=name, frames=frames, fin=fin, cout=a["cout"],
                           cin=a["cin"], h=a["h"], w=a["w"], taps=len(a["taps"]),
                           wiring=wiring, sc_off=a["sc_off"], artifact_index=a["artifact_index"],
                           sa_in=a["sa_in"], sa_out=a["sa_out"],
                           executes=lp["executes"], macs=lp["macs"]))
        emit("CONV", li)
        act = (frames, a["cout"], a["h"], a["w"])
        act_sa = a["sa_out"]
        counts["conv"] = counts.get("conv", 0) + 1
        if shortcut:
            save_slot(1)                     # the residual operand = shortcut output

    def gamma_index(op_json: dict) -> int:
        """`vae_wiring.json` stores `gamma_off` in BYTES (the same trap the
        weight-scale offsets paid for in P3.8a). The device indexes floats."""
        b = int(op_json["gamma_off"])
        assert b % 4 == 0, f"{op_json['site']} gamma_off {b} is not f4-aligned"
        i = b // 4
        assert i + int(op_json["C"]) <= n_gamma, \
            f"{op_json['site']}: floats {i}+{op_json['C']} > {n_gamma}"
        return i

    def next_conv_sa() -> float | None:
        """SA_in of the next conv — the scale a norm/SiLU output is stored at."""
        seen = 0
        for o in ops:
            if o["op"] == "conv":
                seen += 1
                if seen > counts.get("conv", 0):
                    return by_name[o["name"]]["sa_in"]
        return None

    def save_slot(k: int) -> None:
        slot[k] = act
        emit("SAVE", k)
        if act is None:
            raise SystemExit("SAVE before any conv: nothing to save")

    def load_slot(k: int) -> None:
        nonlocal act, act_sa
        if slot[k] is None:
            raise SystemExit(f"LOAD of empty slot {k}")
        act = slot[k]
        emit("LOAD", k)
        # the slot's scale is only known to the device; the symbolic side does
        # not need it (the next op is a norm/SiLU or a conv input build)

    for oi, o in enumerate(ops):
        kind = o["op"]
        if kind == "conv":
            conv_op(o)
        elif kind == "save":
            # The resnet's entry tensor was already captured (before the
            # shortcut conv, or here, where `act` still holds it for a resnet
            # with an identity shortcut). With a shortcut, `act` is now the
            # shortcut's output: that is the residual operand, and norm1 still
            # needs the entry, so reload it.
            alias = str(o.get("alias", ""))
            counts["save"] = counts.get("save", 0) + 1
            if alias.startswith("identity"):
                save_slot(0)
            else:
                load_slot(0)              # the residual operand is already slot 1
        elif kind == "norm":
            out_sa = next_conv_sa()
            if out_sa is None:
                raise SystemExit(f"norm {o['site']} has no consumer conv")
            if oi + 1 < len(ops) and ops[oi + 1]["op"] == "silu":
                pass                          # folded at the SiLU op below
            else:
                emit("NORM", gamma_index(o), int(o["C"]), float(o["scale"]), float(out_sa))
        elif kind == "silu":
            prev = ops[oi - 1]
            out_sa = next_conv_sa()
            if prev["op"] == "norm":
                assert out_sa is not None, "folded norm+SiLU with no consumer conv"
                # one fused pass: L2 norm, gamma, SiLU, quantise at the consumer
                emit("NORMSILU", gamma_index(prev), int(prev["C"]),
                     float(prev["scale"]), float(out_sa))
                counts["norm"] = counts.get("norm", 0) + 1
                counts["silu"] = counts.get("silu", 0) + 1
            else:
                emit("SILU", float(out_sa if out_sa is not None else act_sa))
                counts["silu"] = counts.get("silu", 0) + 1
        elif kind == "add":
            sa = o.get("post_add_sa")
            if sa is None:
                raise SystemExit(f"add {o['site']} has no post_add_sa — recalibrate")
            k = 1 if slot[1] is not None else 0
            emit("ADD", k, float(sa), float(o["post_add_max_abs"]))
            counts["add"] = counts.get("add", 0) + 1
            slot[1] = None
        elif kind in ("temporal_shuffle", "upsample2d"):
            # Folded into the consumer conv's input gather (wiring 1/2/5); the
            # buffer shape does NOT change here, which is what lets the conv's
            # wiring derivation below prove the fold is the right one.
            emit("SKIP", kind)
            counts[kind] = counts.get(kind, 0) + 1
        elif kind == "attn":
            emit("SKIP", "attn")
            counts["attn"] = counts.get("attn", 0) + 1
            logs.append("mid-block attention is NOT executed (no INT8 artifact; "
                        "priced on Mali at 0.0855 s/chunk) — the walk is faithful "
                        "everywhere else, and this is a stated deviation")
        else:
            raise SystemExit(f"unknown op {kind}")

    for k, want in TAPS_WANT.items():
        got = counts.get(k, 0)
        assert got == want, f"op count {k}: got {got}, plan says {want}"

    head = [f"# vae_wire — execution order, artifact_index per conv, {len(ops)} plan ops",
            f"# conv {counts['conv']} norm {counts['norm']} silu {counts['silu']} "
            f"add {counts['add']} save {counts['save']} "
            f"shuffle {counts['temporal_shuffle']} upsample {counts['upsample2d']} "
            f"attn {counts['attn']}"]
    body = head + lines
    for p in prog:
        if p[0] == "CONV":
            body.append(f"OP CONV {p[1]}")
        elif p[0] in ("SAVE", "LOAD"):
            body.append(f"OP {p[0]} {p[1]}")
        elif p[0] == "NORMSILU":
            body.append(f"OP NORMSILU {p[1]} {p[2]} {p[3]:.9g} {p[4]:.9g}")
        elif p[0] == "NORM":
            body.append(f"OP NORM {p[1]} {p[2]} {p[3]:.9g} {p[4]:.9g}")
        elif p[0] == "SILU":
            body.append(f"OP SILU {p[1]:.9g}")
        elif p[0] == "ADD":
            body.append(f"OP ADD {p[1]} {p[2]:.9g} {p[3]:.9g}")
        elif p[0] == "SKIP":
            body.append(f"OP SKIP {p[1]}")

    meta = dict(counts=counts, n_ops=len(ops), sa_in=[min(SAIN), max(SAIN)],
                sa_out=[min(SAOUT), max(SAOUT)],
                executes=sum(L["executes"] for L in layers),
                macs=sum(L["macs"] for L in layers),
                post_add=[o["post_add_sa"] for o in ops if o["op"] == "add"],
                logs=logs)
    return layers, prog, "\n".join(body) + "\n", meta


def parse(stdout: str) -> dict:
    rec = dict(dev=[], probe=[], note=[], compile=[], layer=[], ts=[], summary=[],
               fatal=[], skip=[], done=0)
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        tag, _, rest = line.partition("|")
        try:
            if tag == "CWLAYER":
                d = {}
                for kv in rest.split("|"):
                    k, _, v = kv.partition("=")
                    d[k] = v
                rec["layer"].append(d)
            elif tag == "SUMMARY":
                d = {}
                for kv in rest.split("|"):
                    k, _, v = kv.partition("=")
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
                rec["summary"].append(d)
            elif tag == "TS":
                parts = rest.split("|")
                d = {}
                for kv in parts[1:]:
                    k, _, v = kv.partition("=")
                    d[k] = int(v) if k in ("l", "t", "f") else float(v)
                rec["ts"].append(d)
            elif tag in ("SEQ", "WIRE", "PROBE", "NOTE", "CSCOMPILE", "CONVWIRE"):
                key = "compile" if tag == "CSCOMPILE" else tag.lower()
                rec.setdefault(key, []).append(line)
            elif tag == "DEV":
                rec["dev"].append(line)
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
    ap.add_argument("--convs", type=int, default=0, help="0 = all 36 convs")
    ap.add_argument("--pc", type=int, default=-1, choices=[-1, 0, 1])
    ap.add_argument("--cooldown", action="store_true",
                    help="refuse to run unless max(TPU, skin, VIRTUAL-SKIN) <= battery + 3")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="vae_wire")
    args = ap.parse_args()

    layers, prog, text, meta = build_wire()
    PLAN_LOCAL.write_text(text, encoding="utf-8")
    print(f"wire: {len(layers)} convs, {len(prog)} program ops, "
          f"{meta['executes']} CONV_2D executes -> {PLAN_LOCAL}")
    print(f"  op counts {meta['counts']}")
    print(f"  SA_in  {meta['sa_in'][0]:.6g} .. {meta['sa_in'][1]:.6g}   "
          f"SA_out {meta['sa_out'][0]:.6g} .. {meta['sa_out'][1]:.6g}")
    for L in layers:
        print(f"  [{L['idx']:2d}] ai={L['artifact_index']:2d} {L['name'][:44]:44s} "
              f"f={L['frames']} {L['cin']}x{L['h']}x{L['w']}->{L['cout']} "
              f"w={L['wiring']} sa={L['sa_in']:.4g}/{L['sa_out']:.4g}")
    for m in meta["logs"]:
        print(f"  NOTE {m}")
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
                     (VAE_DIR / "act_scales2_f32.bin", R_SA2),
                     (VAE_DIR / "norms_f32.bin", R_NM),
                     (PLAN_LOCAL, R_WIRE),
                     (BIN, REMOTE_BIN)):
        rc, txt = run([str(ADB), "-s", serial, "push", str(src), dst], timeout=300)
        if rc != 0:
            print(f"push failed for {src}: {txt}", file=sys.stderr)
            return 2
        print(f"pushed {src.name} -> {dst}")
    adb(serial, f"shell chmod 755 {REMOTE_BIN}")

    adb(serial, "logcat -c")
    cmd = f"shell {REMOTE_BIN} convwire {args.convs} {args.pc}"
    print(f"running: {cmd}")
    rc, out = adb(serial, cmd, timeout=args.timeout)
    print(out)
    zones1 = read_zones(serial)
    _, log = adb(serial, "logcat -d -s NeuralNetworks:* NnapiHal:* nnapi:* -t 160", timeout=60)

    rec = parse(out)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    doc = dict(serial=serial, timestamp=ts, cmd=cmd, rc=rc,
               bin_sha=sha16(BIN), plan_sha=hashlib.sha256(text.encode()).hexdigest()[:16],
               weights_sha=sha16(VAE_DIR / "weights_int8.bin"),
               sa2_sha=sha16(VAE_DIR / "act_scales2_f32.bin"),
               norms_sha=sha16(VAE_DIR / "norms_f32.bin"),
               convs=args.convs, pc=args.pc, layers_planned=layers, meta=meta,
               thermal=dict(before=zones0, after=zones1, gate=gate, verdict=verdict),
               parsed=rec, logcat_tail=log[-4000:])
    js = DEVICE_DIR / f"{args.out}_{ts}.json"
    js.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    s = (rec["summary"] or [{}])[0]
    md = [f"# VAE decode, wired (`convwire`) — {ts}", "",
          f"serial `{serial}` · binary `{doc['bin_sha']}` · weights `{doc['weights_sha']}` · "
          f"SA2 `{doc['sa2_sha']}` · norms `{doc['norms_sha']}` · "
          f"convs={args.convs or 'all'} pc={args.pc} rc={rc}", "", "## Thermal", "",
          "- before: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(zones0.items()) if k in ZONE_KEYS),
          "- after:  " + " ".join(f"{k}={v:.1f}" for k, v in sorted(zones1.items()) if k in ZONE_KEYS),
          f"- gate max(TPU,skin,VIRTUAL-SKIN) <= battery+3 = {gate} -> **{verdict}**", ""]
    if s:
        md += ["## Result", "", "| quantity | value |", "|---|---|",
               f"| convs executed | {s.get('convs')} |",
               f"| `CONV_2D` executes | **{s.get('executes')}** |",
               f"| conv execute walls | **{s.get('conv_ms')} ms** |",
               f"| norm+SiLU passes | {s.get('silk_ms')} ms |",
               f"| residual adds | {s.get('add_ms')} ms |",
               f"| input build + quantise | {s.get('prep_ms')} ms |",
               f"| output dequant | {s.get('epi_ms')} ms |",
               f"| compile | {s.get('compile_s')} s |",
               f"| wall (whole walk) | **{s.get('wall_s')} s** |",
               f"| MACs | {s.get('macs')} |",
               f"| per-channel used | {s.get('pc')} (fallbacks {s.get('pc_fallbacks')}) |",
               f"| op saturation (clamped) | {s.get('clamp')} % |",
               f"| **sum of execute walls (TS)** | **{sum(t['ms'] for t in rec['ts']):.1f} ms** over "
               f"{len(rec['ts'])} executes ({sum(t['ms'] for t in rec['ts']) / max(1, len(rec['ts'])):.2f} ms mean, "
               f"{max((t['ms'] for t in rec['ts']), default=0):.2f} ms max) |",
               f"| norm+SiLU elements | {s.get('el_norm')} |",
               f"| residual-add elements | {s.get('el_add')} |", ""]
    if rec["layer"]:
        md += ["## Per conv (execution order)", "",
               "| li | artifact | layer | frames | taps | execs | conv ms | prep ms | "
               "epi ms | GOPS | SA_in | SA_out | clamp % |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for L in rec["layer"]:
            md.append(f"| {L.get('layer')} | {L.get('ai')} | `{L.get('name')}` | {L.get('frames')} | "
                      f"{L.get('taps')} | {L.get('executes')} | {L.get('conv_ms')} | {L.get('prep_ms')} | "
                      f"{L.get('epi_ms')} | {L.get('gops')} | {L.get('sa_in')} | {L.get('sa_out')} | "
                      f"{L.get('clamp')} |")
        md.append("")
    if rec["note"]:
        md += ["## Notes", "", "```"] + rec["note"] + ["```", ""]
    if rec["fatal"]:
        md += ["## Failures", "", "```"] + rec["fatal"] + ["```", ""]
    md += ["<details><summary>logcat (NNAPI)</summary>", "", "```", log[-2500:], "```", "",
           "</details>", ""]
    mdp = DEVICE_DIR / f"{args.out}_{ts}.md"
    mdp.write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {js}")
    print(f"wrote {mdp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
