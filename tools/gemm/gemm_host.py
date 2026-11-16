#!/usr/bin/env python3
"""Mali INT8 GEMM benchmark — host driver (Phase 2/3, Gate 2 input).

Builds `tools/android/gemm`, pushes the binary + kernel source to the phone,
runs it, and records every number to docs/device/gemm_mali_<ts>.{json,md}.

    python tools/gemm/gemm_host.py --build
    python tools/gemm/gemm_host.py --mode all --iters 5
    python tools/gemm/gemm_host.py --mode block --M 1530 --variants tiled_64x64x32

Thermal protocol: the run is gated on max(SoC zone) <= ambient + 3 C, and the
zones are recorded before and after so a throttled result is visible rather
than silently reported. Nothing in here models a rate — the numbers are the
phone's.
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
from probe_host import ADB, adb, cooldown, pick_serial, read_zones, run  # noqa: E402

GEMM_DIR = REPO / "tools" / "android" / "gemm"
BIN = GEMM_DIR / "build" / "gemm_bench"
KERNEL = GEMM_DIR / "kernels" / "gemm8.cl"
REMOTE_BIN = "/data/local/tmp/gemm_bench"
REMOTE_CL = "/data/local/tmp/gemm8.cl"
DEVICE_DIR = REPO / "docs" / "device"

ZONE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def parse(stdout: str) -> dict:
    rec = {"device": {}, "builds": [], "verify": [], "gemm": [], "block": [],
           "f16": [], "conv": [], "conv_verify": [], "tv": None, "fatal": []}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith("=="):
            continue
        if line.startswith("CL_DEVICE "):
            for kv in line[len("CL_DEVICE "):].split():
                k, _, v = kv.partition("=")
                rec["device"][k] = v
            continue
        if line.startswith("FATAL"):
            rec["fatal"].append(line)
            continue
        parts = line.split("|")
        tag = parts[0]
        try:
            if tag == "BUILD":
                rec["builds"].append({"variant": parts[1], "status": parts[2],
                                      "detail": parts[3:]})
            elif tag == "VERIFY" and parts[0] == "VERIFY" and len(parts) > 3:
                d = {"variant": parts[1]}
                for kv in parts[2:]:
                    k, sep, v = kv.partition("=")
                    if sep:
                        d[k] = v
                    else:
                        d["result"] = k          # MATCH / MISMATCH token
                rec["verify"].append(d)
            elif tag == "GEMM":
                d = {"variant": parts[1], "shape": parts[2]}
                for kv in parts[3:]:
                    k, _, v = kv.partition("=")
                    d[k] = float(v) if k in ("ms", "gops") else int(v)
                rec["gemm"].append(d)
            elif tag == "BLOCK":
                d = {"variant": parts[1]}
                for kv in parts[2:]:
                    k, _, v = kv.partition("=")
                    d[k] = float(v) if k in ("ms", "gops", "gmac") else int(v)
                rec["block"].append(d)
            elif tag == "F16":
                d = {"variant": parts[1]}
                for kv in parts[2:]:
                    k, _, v = kv.partition("=")
                    d[k] = v
                rec["f16"].append(d)
            elif tag == "CONV":
                d = {"class": parts[1]}
                for kv in parts[2:]:
                    k, sep, v = kv.partition("=")
                    if not sep:
                        continue
                    if k in ("H", "W", "Cin", "CinP", "Cout", "taps", "s", "frames", "ops"):
                        d[k] = int(v)
                    else:
                        try:
                            d[k] = float(v)
                        except ValueError:
                            d[k] = v
                rec["conv"].append(d)
            elif tag == "CONVV":
                d = {"class": parts[1], "result": parts[2]}
                for kv in parts[3:]:
                    k, sep, v = kv.partition("=")
                    if sep:
                        d[k] = v
                rec["conv_verify"].append(d)
            elif tag == "TV_MALI":
                d = {}
                for kv in parts[1:]:
                    k, _, v = kv.partition("=")
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
                rec["tv"] = d
            elif tag == "VERIFY_SUMMARY":
                rec["verify_summary"] = parts[1]
        except (IndexError, ValueError):
            continue
    return rec


def md_report(rec: dict) -> str:
    ts = rec["timestamp"]
    L = [f"# Mali-G78 INT8 GEMM — {ts}", "",
         f"serial `{rec['serial']}` · kernel `{rec['kernel_sha256']}` · "
         f"binary `{rec['binary_sha256']}` · mode `{rec['mode']}` iters={rec['iters']}", ""]
    dev = rec["parsed"]["device"]
    L += ["## Device", "",
          f"- `{dev.get('name','?')}` · CUs **{dev.get('cus','?')}** · "
          f"driver-reported max clock `{dev.get('max_clock_mhz','?')} MHz`", ""]
    tb, ta = rec["temps"]["before"], rec["temps"]["after"]
    L += ["## Thermal", "",
          "- before: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(tb.items()) if k in ZONE_KEYS),
          "- after:  " + " ".join(f"{k}={v:.1f}" for k, v in sorted(ta.items()) if k in ZONE_KEYS),
          ""]
    p = rec["parsed"]
    if p["verify"]:
        L += ["## Exactness (int32, vs naive reference)", "",
              "| variant | M | N | K | result | mismatch | ms |", "|---|---|---|---|---|---|---|"]
        for v in p["verify"]:
            L.append(f"| `{v['variant']}` | {v.get('M')} | {v.get('N')} | {v.get('K')} | "
                     f"**{v.get('result','?')}** | {v.get('mismatch')} | {v.get('ms','')} |")
        L.append("")
    if p["gemm"]:
        L += ["## GOPS sweep", "",
              "| variant | shape | M | N | K | ms | GOPS |", "|---|---|---|---|---|---|---|"]
        for g in p["gemm"]:
            L.append(f"| `{g['variant']}` | {g['shape']} | {g['M']} | {g['N']} | {g['K']} | "
                     f"{g['ms']:.3f} | **{g['gops']:.1f}** |")
        L.append("")
    if p["block"]:
        L += ["## DiT block (10 GEMMs, timed as one sequence)", "",
              "| variant | M | lora | ms/block | GOPS | GMAC |", "|---|---|---|---|---|---|"]
        for b in p["block"]:
            L.append(f"| `{b['variant']}` | {b['M']} | {b['lora']} | **{b['ms']:.1f}** | "
                     f"{b['gops']:.1f} | {b['gmac']:.2f} |")
        L.append("")
    if p["f16"]:
        L += ["## FP16 GEMM (LoRA-up path)", "", "| variant | M | N | K | ms | GOPS |",
              "|---|---|---|---|---|---|"]
        for f in p["f16"]:
            L.append(f"| `{f['variant']}` | {f.get('M')} | {f.get('N')} | {f.get('K')} | "
                     f"{f.get('ms')} | **{f.get('gops')}** |")
        L.append("")
    if p["conv"]:
        L += ["## VAE decode conv classes (implicit GEMM, one dispatch per class)", "",
              "| class | HxW | Cin | Cout | taps | s | frames | ops | ms/frame | GOPS | layer ms | class ms |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for g in p["conv"]:
            L.append(f"| `{g['class']}` | {g.get('H','')}x{g.get('W','')} | {g.get('Cin','')} | "
                     f"{g.get('Cout','')} | {g.get('taps','')} | {g.get('s','')} | {g.get('frames','')} | "
                     f"{g.get('ops','')} | {g.get('ms', 0):.3f} | **{g.get('gops', 0):.1f}** | "
                     f"{g.get('layer_ms', 0):.1f} | **{g.get('cls_ms', 0):.1f}** |")
        L.append("")
        if p["tv"]:
            t = p["tv"]
            L += [f"**T_V (Mali) = {t.get('total_s', 0):.3f} s/chunk**, "
                  f"{t.get('gops', 0):.1f} GOPS aggregate, {t.get('tmacs', 0):.1f} GMAC, "
                  f"{int(t.get('classes', 0))} classes", ""]
        if p["conv_verify"]:
            L += ["### Exactness vs the int32 scalar reference", "",
                  "| class | result | mismatch | ms (reference) |", "|---|---|---|---|"]
            for v in p["conv_verify"]:
                L.append(f"| `{v['class']}` | **{v['result']}** | {v.get('mismatch','')} | "
                         f"{v.get('ms_ref','')} |")
            L.append("")
    L += ["## Build log", "", "```"]
    for b in p["builds"]:
        L.append(f"{b['variant']}: {b['status']} {' '.join(b['detail'])}")
    for f in p["fatal"]:
        L.append(f)
    L += ["```", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--mode", default="all",
                    choices=["verify", "bench", "f16", "block", "peak", "diag", "conv", "all"])
    ap.add_argument("--variants", default="", help="comma-separated variant filter")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--M", type=int, default=1530, help="block-mode token count")
    ap.add_argument("--cooldown", type=float, default=None,
                    help="gate on max(SoC) <= this C before the timed run")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    if not ADB.exists():
        print(f"adb missing at {ADB}", file=sys.stderr)
        return 2

    if args.build:
        print("building gemm_bench ...")
        rc, txt = run(["bash", (GEMM_DIR / "build.sh").as_posix()], timeout=900)
        print(txt[-1500:])
        if rc != 0:
            return rc
    if not BIN.exists():
        print("gemm_bench not built — rerun with --build", file=sys.stderr)
        return 2

    serial = args.serial or pick_serial()
    if not serial:
        print("no device", file=sys.stderr)
        return 2

    if args.cooldown is not None:
        if cooldown(serial, args.cooldown) != 0:
            print("cooldown gate not satisfied — refusing to report a throttled run",
                  file=sys.stderr)
            return 3

    for local, remote in ((BIN, REMOTE_BIN), (KERNEL, REMOTE_CL)):
        rc, txt = run([str(ADB), "-s", serial, "push", str(local), remote], timeout=180)
        if rc != 0:
            print(f"push failed {local}: {txt}", file=sys.stderr)
            return 2
    adb(serial, f"shell chmod 755 {REMOTE_BIN}")

    temps_before = read_zones(serial)
    cmd = f"shell {REMOTE_BIN} {args.mode} {REMOTE_CL} {args.variants} {args.iters} {args.M}"
    print(f"running: {cmd}  (timeout {args.timeout}s)")
    rc, out = adb(serial, cmd, timeout=args.timeout)
    temps_after = read_zones(serial)
    print(out)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    rec = {"serial": serial, "timestamp": ts, "mode": args.mode, "iters": args.iters,
           "M_block": args.M, "rc": rc, "kernel_sha256": sha256(KERNEL),
           "binary_sha256": sha256(BIN), "temps": {"before": temps_before, "after": temps_after},
           "parsed": parse(out)}
    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    js = DEVICE_DIR / f"gemm_mali_{ts}.json"
    js.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    md = DEVICE_DIR / f"gemm_mali_{ts}.md"
    md.write_text(md_report(rec), encoding="utf-8")
    print(f"\nwrote {js}\nwrote {md}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
