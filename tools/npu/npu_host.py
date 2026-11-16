#!/usr/bin/env python3
"""T_N — token-tiled NNAPI block graph on the Tensor Edge TPU (Phase 3 input).

Builds `tools/android/npublock`, pushes it, and measures one Preset-A DiT block
(10 INT8 GEMMs) tiled 4x on the token axis at M=384 — the measured per-graph
token ceiling from Gate 0.

    python tools/npu/npu_host.py --build
    python tools/npu/npu_host.py --modes support,chain --tiles 4 --iters 3
    python tools/npu/npu_host.py --modes support --M 384      # validate only

Protocol:
  * `adb logcat -c` before any NNAPI work, and logcat is captured afterwards so
    the NNAPI manager's own validator message is in the artifact.
  * One fresh process per mode (`adb shell` per mode): DEAD_OBJECT is sticky
    within a session, so a failed config must not poison the next one.
  * Nothing is modelled. A rate that was not measured stays null.
  * Thermal zones recorded before and after; a throttled sample is visible
    rather than silently averaged in.

Weights come from ONE file-mapped INT8 blob on the device (46,399,488 B = the
block's 10 base matrices) served to NNAPI via createFromFd — route B, the same
one-page-cache artifact the GPU imports. The blob is a deterministic LCG fill:
this measures *latency*, which is shape-determined, and the real INT8 artifact
does not exist yet (Phase 1 quantiser outstanding).
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
from probe_host import ADB, adb, pick_serial, read_zones, run  # noqa: E402

NPU_DIR = REPO / "tools" / "android" / "npublock"
BIN = NPU_DIR / "build" / "npu_block"
REMOTE_BIN = "/data/local/tmp/npu_block"
DEVICE_DIR = REPO / "docs" / "device"

ZONE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")
BLOCK_MACS_PER_TOKEN = 46_399_488
FATAL_STOP = ("OP_FAILED", "DEAD_OBJECT", "BAD_STATE")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def uptime(serial: str) -> float | None:
    rc, out = adb(serial, "shell cat /proc/uptime", timeout=30)
    try:
        return float(out.split()[0]) if rc == 0 and out.strip() else None
    except (ValueError, IndexError):
        return None


def parse(stdout: str) -> dict:
    rec = {"dev": [], "support": [], "sup_op": [], "model_finish": [], "compile": [],
           "tile": [], "op": [], "summary": [], "fatal": [], "wfile": [], "done": 0}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        tag = parts[0]
        try:
            if tag == "DEV":
                d = dict(kv.split("=", 1) for kv in parts[1:] if "=" in kv)
                rec["dev"].append(d)
            elif tag == "WFILE":
                rec["wfile"].append(dict(kv.split("=", 1) for kv in parts[1:] if "=" in kv))
            elif tag == "SUPPORT":
                d = dict(kv.split("=", 1) for kv in parts[1:] if "=" in kv)
                rec["support"].append(d)
            elif tag == "SUP_OP":
                rec["sup_op"].append({"pc": parts[1], "op": parts[2], "supported": int(parts[3])})
            elif tag == "BUILD_FAIL":
                rec["fatal"].append(line)
            elif tag == "MODEL_FINISH":
                d = dict(kv.split("=", 1) for kv in parts[1:] if "=" in kv)
                rec["model_finish"].append(d)
            elif tag == "COMPILE":
                d = dict(kv.split("=", 1) for kv in parts[1:] if "=" in kv)
                rec["compile"].append(d)
            elif tag == "TILE":
                d = {"tag": parts[1]}
                for kv in parts[2:]:
                    k, _, v = kv.partition("=")
                    d[k] = float(v) if k.startswith("ms") else v
                rec["tile"].append(d)
            elif tag == "OP":
                d = {"op": parts[1]}
                for kv in parts[2:]:
                    k, _, v = kv.partition("=")
                    d[k] = float(v) if k in ("compile_ms", "ms_per_tile", "per_block_s", "per_pass_s", "gops") else int(v)
                rec["op"].append(d)
            elif tag == "SUMMARY":
                d = {}
                for kv in parts[1:]:
                    k, _, v = kv.partition("=")
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
                rec["summary"].append(d)
            elif tag == "FATAL":
                rec["fatal"].append(line)
            elif tag == "DONE":
                rec["done"] += 1
        except (IndexError, ValueError):
            continue
    return rec


def md_report(rec: dict) -> str:
    ts = rec["timestamp"]
    L = [f"# T_N — token-tiled NNAPI block graph — {ts}", "",
         f"serial `{rec['serial']}` · binary `{rec['binary_sha256']}` · "
         f"modes `{','.join(rec['modes'])}` · tiles={rec['tiles']} iters={rec['iters']} M={rec['M']}", ""]

    L += ["## Thermal (gate: max(SoC) <= ambient + 3 C)", ""]
    tb, ta = rec["temps"]["before"], rec["temps"]["after"]
    L.append("- before: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(tb.items()) if k in ZONE_KEYS))
    L.append("- after:  " + " ".join(f"{k}={v:.1f}" for k, v in sorted(ta.items()) if k in ZONE_KEYS))
    L.append("")

    for name, p in rec["runs"].items():
        L += [f"## mode `{name}` (rc={p['rc']})", ""]
        if p.get("device_rebooted"):
            L += [f"> **DEVICE REBOOTED during this mode** — uptime "
                  f"{p.get('uptime_before')} s -> {p.get('uptime_after')} s. "
                  "The run is truncated; no rate from it.", ""]
        if p["dev"]:
            for d in p["dev"]:
                L.append(f"- device[{d.get('idx')}] `{d.get('name')}` type={d.get('type')}")
            L.append("")
        if p["wfile"]:
            w = p["wfile"][0]
            L.append(f"- weight blob `{w.get('path')}` {int(w.get('bytes', 0)):,} B (file-mapped, createFromFd)")
            L.append("")
        if p["sup_op"]:
            L += ["| op | scales | Edge-TPU support |", "|---|---|---|"]
            for s in p["sup_op"]:
                L.append(f"| `{s['op']}` | {s['pc']} | {'yes' if s['supported'] else '**NO**'} |")
            L.append("")
        if p["model_finish"]:
            for m in p["model_finish"]:
                L.append(f"- model_finish: nops={m.get('nops')} M={m.get('M')} rc={m.get('rc')}")
            L.append("")
        if p["compile"]:
            for c in p["compile"]:
                L.append(f"- compile: {c.get('ms')} ms rc={c.get('rc')}")
            L.append("")
        for s in p["summary"]:
            L += ["| quantity | value |", "|---|---|"]
            mode = s.get("mode")
            if s.get("probe_only"):
                L.append(f"| compile-only probe (`{mode}`, M={s.get('M')}) | **compiled OK in {s.get('compile_ms')} ms** |")
                L.append("")
                continue
            if mode == "chain":
                L.append(f"| compile (one 10-op graph) | {s.get('compile_ms')} ms |")
                L.append(f"| compute per token tile (M=384) | **{s.get('ms_per_tile')} ms** |")
                L.append(f"| **T_N per block** ({int(s.get('tiles', 0))} tiles) | **{s.get('per_block_s')} s** |")
                L.append(f"| effective | **{s.get('gops')} GOPS** |")
                L.append("")
            elif str(mode).startswith("down"):
                L.append(f"| compile (chunked-K graph) | {s.get('compile_ms')} ms |")
                L.append(f"| compute per token tile (M={s.get('M')}) | **{s.get('ms_per_tile')} ms** |")
                L.append(f"| **`ffn.down` per block** ({int(s.get('tiles', 0))} tiles) | **{s.get('per_block_s')} s** |")
                L.append(f"| effective | **{s.get('gops')} GOPS** |")
                L.append("")
            else:
                L.append(f"| compile (10 graphs) | {s.get('compile_ms')} ms |")
                L.append(f"| **T_N per block** ({int(s.get('tiles', 0))} tiles) | **{s.get('per_block_s')} s** |")
                L.append(f"| effective | **{s.get('gops')} GOPS** |")
                L.append("")
        if p["op"]:
            L += ["| op | N | K | compile ms | ms/tile | s/block | GOPS |", "|---|---|---|---|---|---|---|"]
            for o in p["op"]:
                per = o.get("per_pass_s", o.get("per_block_s"))
                L.append(f"| `{o['op']}` | {o.get('N')} | {o.get('K')} | {o.get('compile_ms')} | "
                         f"{o.get('ms_per_tile')} | {per} | {o.get('gops')} |")
            L.append("")
        if p["fatal"]:
            L += ["```"] + p["fatal"] + ["```", ""]
        if p.get("logcat"):
            L += ["<details><summary>logcat (NNAPI)</summary>", "", "```",
                  p["logcat"][-2500:], "```", "", "</details>", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--modes", default="support,chain",
                    help="comma list of support|chain|single|down2|down4 (one fresh process each). "
                         "down2/down4 = chunked-K ffn.down workaround; with --iters 0 they are "
                         "compile-only probes.")
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--M", type=int, default=384)
    ap.add_argument("--pc", type=int, default=0, choices=[0, 1],
                    help="1 = per-output-column weight scales (SYMM_PER_CHANNEL), "
                         "0 = per-tensor. Measured: per-column is rejected at addOperation "
                         "on this HAL, so 0 is the only compilable path.")
    ap.add_argument("--cooldown", type=float, default=None,
                    help="gate on max(SoC) <= this C before the timed run")
    ap.add_argument("--timeout", type=int, default=420, help="per-mode device timeout")
    ap.add_argument("--only", type=int, default=-1,
                    help="single-mode: run only this GEMM index (0-9), -1 = all 10")
    ap.add_argument("--out", default="npu_tn", help="artifact filename prefix (docs/device/<out>_<ts>)")
    ap.add_argument("--serial", default=None)
    args = ap.parse_args()

    if not ADB.exists():
        print(f"adb missing at {ADB}", file=sys.stderr)
        return 2
    if args.build:
        print("building npu_block ...")
        rc, txt = run(["bash", str(NPU_DIR / "build.sh").replace("\\", "/")], timeout=900)
        print(txt[-1200:])
        if rc != 0:
            return rc
    if not BIN.exists():
        print("npu_block not built — rerun with --build", file=sys.stderr)
        return 2

    serial = args.serial or pick_serial()
    if not serial:
        print("no device", file=sys.stderr)
        return 2

    if args.cooldown is not None:
        zones = read_zones(serial)
        peak = max((v for k, v in zones.items() if k in ZONE_KEYS), default=None)
        print(f"thermal gate: peak={peak} C wants <= {args.cooldown} C")
        if peak is not None and peak > args.cooldown:
            print("TOO WARM — refusing to report a throttled run", file=sys.stderr)
            return 3

    rc, txt = run([str(ADB), "-s", serial, "push", str(BIN), REMOTE_BIN], timeout=180)
    if rc != 0:
        print(f"push failed: {txt}", file=sys.stderr)
        return 2
    adb(serial, f"shell chmod 755 {REMOTE_BIN}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    temps_before = read_zones(serial)
    runs: dict[str, dict] = {}
    stop = False

    for mode in modes:
        if stop:
            runs[mode] = {"rc": -1, "skipped": "earlier mode took the HAL down", "dev": [],
                          "support": [], "sup_op": [], "model_finish": [], "compile": [],
                          "tile": [], "op": [], "summary": [], "fatal": [], "wfile": []}
            continue
        adb(serial, "logcat -c")
        iters = 2 if mode == "single" else args.iters
        up0 = uptime(serial)
        cmd = f"shell {REMOTE_BIN} {mode} {args.tiles} {iters} {args.M} {args.pc} {args.only}"
        print(f"running: {cmd}")
        rc, out = adb(serial, cmd, timeout=args.timeout)
        print(out)
        up1 = uptime(serial)
        _, log = adb(serial, "logcat -d -s NeuralNetworks:* NnapiHal:* nnapi:* -t 120", timeout=60)
        p = parse(out)
        p["rc"] = rc
        p["logcat"] = log
        p["uptime_before"], p["uptime_after"] = up0, up1
        # A reboot mid-mode means the HAL took the device down: report it, do not
        # silently record a truncated run as if it had finished.
        p["device_rebooted"] = (up0 is not None and up1 is not None and up1 < up0)
        runs[mode] = p
        if p["device_rebooted"]:
            print(f"  -> DEVICE REBOOTED during mode {mode} (uptime {up0:.0f}s -> {up1:.0f}s)")
            stop = True
        if any(tok in f for f in p["fatal"] for tok in FATAL_STOP):
            print(f"  -> HAL-harming failure in mode {mode}; not submitting further graphs")
            stop = True

    temps_after = read_zones(serial)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    rec = {"serial": serial, "timestamp": ts, "modes": modes, "tiles": args.tiles,
           "iters": args.iters, "M": args.M, "pc": args.pc, "binary_sha256": sha256(BIN),
           "temps": {"before": temps_before, "after": temps_after}, "runs": runs}
    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    js = DEVICE_DIR / f"{args.out}_{ts}.json"
    js.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    md = DEVICE_DIR / f"{args.out}_{ts}.md"
    md.write_text(md_report(rec), encoding="utf-8")
    print(f"\nwrote {js}\nwrote {md}")

    for m, p in runs.items():
        for s in p.get("summary", []):
            print(f"  {m}: T_N per block = {s.get('per_block_s')} s  ({s.get('gops')} GOPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
