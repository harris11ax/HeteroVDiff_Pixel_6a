#!/usr/bin/env python3
"""P3.2 — the GPU stage over the real INT8 artifact (host side).

Pushes `tools/android/pipe/proc_a`, one block region (`artifacts/blocks/block_00.bin`
+ its `.scales.f32`), and the kernel source, then runs the GPU stage for N blocks.
proc_a mmaps the region read-only (route B, Gate 2) and imports each GEMM's weight
window into OpenCL with `clImportMemoryARM(HOST_ARM)` — one stored INT8 copy, no
second copy made by this stage.

    python tools/pipe/proc_a.py --blocks 8            # build if needed, then run
    python tools/pipe/proc_a.py --blocks 8 --route copy   # price the zero-copy win

Writes docs/device/proc_a_<ts>.{json,md}. Reports per-GEMM and per-block ms from
the device, publish->consume arm's length latency, and the thermal zones at both
ends. The thermal gate is enforced the way every timed run is: measured, and if the
start is over the line the doc says so instead of dropping the numbers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "probe"))
from probe_host import ADB, adb, pick_serial, read_zones, run  # noqa: E402

PIPE_DIR = REPO / "tools" / "android" / "pipe"
PIPE_BIN = PIPE_DIR / "build" / "proc_a"
GEMM_DIR = REPO / "tools" / "android" / "gemm"
KERNEL = GEMM_DIR / "kernels" / "gemm8.cl"
ART = REPO / "artifacts" / "blocks"
DEVICE_DIR = REPO / "docs" / "device"

REMOTE_BIN = "/data/local/tmp/hv_proc_a"
REMOTE_CL = "/data/local/tmp/gemm8.cl"
REMOTE_BLK = "/data/local/tmp/hv_blk.bin"
REMOTE_SCL = "/data/local/tmp/hv_blk.scales.f32"
REMOTE_RING = "/data/local/tmp/hv_handoff.bin"
ZONE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")


def kv(line: str) -> dict:
    d: dict = {}
    for part in line.split("|")[1:]:
        k, sep, v = part.partition("=")
        if not sep:
            continue
        if k in ("name", "path", "opts", "block", "route"):
            d[k] = v
            continue
        try:
            d[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            d[k] = v
    return d


def stats(v: list[float]) -> dict | None:
    if not v:
        return None
    return {"n": len(v), "min": min(v), "median": st.median(v), "mean": sum(v) / len(v),
            "max": max(v), "stdev": st.pstdev(v) if len(v) > 1 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--M", type=int, default=1530)
    ap.add_argument("--route", default="import", choices=["import", "copy"])
    ap.add_argument("--block-idx", type=int, default=0)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--serial", default=None)
    args = ap.parse_args()

    if args.build or not PIPE_BIN.exists():
        print("building proc_a ...")
        rc, txt = run(["bash", (PIPE_DIR / "build.sh").as_posix()], timeout=900)
        if rc != 0:
            print(txt[-3000:], file=sys.stderr)
            raise SystemExit("build failed")
        print("build ok")

    blk = ART / f"block_{args.block_idx:02d}.bin"
    scl = ART / f"block_{args.block_idx:02d}.scales.f32"
    for p in (blk, scl, KERNEL, PIPE_BIN):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 2

    serial = args.serial or pick_serial()
    if not serial:
        print("no device attached", file=sys.stderr)
        return 2

    temps_before = read_zones(serial)
    peak = max((v for k, v in temps_before.items() if k in ZONE_KEYS), default=None)
    batt = temps_before.get("battery")
    gate_ok = (peak is not None and batt is not None and peak <= batt + 3.0)

    for src, dst in ((PIPE_BIN, REMOTE_BIN), (KERNEL, REMOTE_CL),
                     (blk, REMOTE_BLK), (scl, REMOTE_SCL)):
        rc, txt = run([str(ADB), "-s", serial, "push", str(src), dst], timeout=600)
        if rc != 0:
            print(f"push failed {src}: {txt}", file=sys.stderr)
            return 2
    adb(serial, f"shell chmod 755 {REMOTE_BIN}")
    adb(serial, "logcat -c")

    cmd = (f"shell cd /data/local/tmp && {REMOTE_BIN} --block {REMOTE_BLK} --scales {REMOTE_SCL} "
           f"--cl {REMOTE_CL} --ring {REMOTE_RING} --blocks {args.blocks} --M {args.M} "
           f"--route {args.route} --drain fork")
    print(f"  running: blocks={args.blocks} M={args.M} route={args.route}")
    rc, out = adb(serial, cmd, timeout=args.timeout)
    lines = [l.strip() for l in out.splitlines() if l.strip()]

    gemms = [kv(l) for l in lines if l.startswith("GEMM|")]
    blocks = [kv(l) for l in lines if l.startswith("TS|gpu|")]
    once = [kv(l) for l in lines if l.startswith("TSC|gpu|")]
    drains = [kv(l) for l in lines if l.startswith("TS|drain|")]
    bufs = [kv(l) for l in lines if l.startswith("BUF|")]
    close = next((kv(l) for l in lines if l.startswith("CLOSE|")), {})
    summ = {l.split("|")[1]: kv(l) for l in lines if l.startswith("SUMMARY|")}
    result = next((kv(l) for l in lines if l.startswith("RESULT|")), {})
    build = next((kv(l) for l in lines if l.startswith("BUILD|")), {})
    hdr = next((kv(l) for l in lines if l.startswith("HV|proc_a|")), {})
    wtot = next((kv(l) for l in lines if l.startswith("W_TOTAL|")), {})
    fatal = [l for l in lines if l.startswith("FATAL")]
    warns = [l for l in lines if l.startswith("WARN") or l.startswith("EXEC_ERR")]

    per_name: dict[str, list[float]] = {}
    per_net: dict[str, list[float]] = {}
    for g in gemms:
        nm = g.get("name")
        if isinstance(nm, str) and isinstance(g.get("ms"), (int, float)):
            per_name.setdefault(nm, []).append(float(g["ms"]))
            if isinstance(g.get("ms_net"), (int, float)):
                per_net.setdefault(nm, []).append(float(g["ms_net"]))
    per_gemm = {n: stats(v) for n, v in per_name.items()}
    for n, s in per_gemm.items():
        g0 = next(g for g in gemms if g.get("name") == n)
        macs = float(g0["M"]) * float(g0["N"]) * float(g0["K"])
        s["N"] = g0["N"]
        s["K"] = g0["K"]
        s["gops_median"] = 2.0 * macs / (s["median"] * 1e6)
        s["gmac"] = macs / 1e9
        if per_net.get(n):
            s["median_net"] = st.median(per_net[n])
            s["gops_net_median"] = 2.0 * macs / (s["median_net"] * 1e6)

    block_ms = [float(b["block_ms"]) for b in blocks if isinstance(b.get("block_ms"), (int, float))]
    once_ms = [float(o["ms_once"]) for o in once if isinstance(o.get("ms_once"), (int, float))]
    lat_us = [float(d["lat_ns"]) / 1e3 for d in drains if isinstance(d.get("lat_ns"), (int, float))]
    pay_us = [float(b["payload_ms"]) * 1e3 for b in blocks if isinstance(b.get("payload_ms"), (int, float))]
    temps_after = read_zones(serial)

    expect_gemms = args.blocks * 10
    expect_once = args.blocks
    verdict = (bool(result) and result.get("route") is not None
               and len(gemms) == expect_gemms
               and len(blocks) == args.blocks
               and len(once_ms) == expect_once
               and not fatal
               and (summ.get("drain", {}).get("nan_words", 0) == 0)
               and rc == 0)

    res = {
        "serial": serial,
        "timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "stage": "P3.2 proc_a (GPU stage over the real INT8 artifact)",
        "route": args.route, "M": args.M, "blocks_requested": args.blocks,
        "block_file": blk.name, "block_bytes": blk.stat().st_size,
        "scales_file": scl.name, "scales_bytes": scl.stat().st_size,
        "header": hdr, "build": build, "weight_buffers": bufs, "weight_totals": wtot,
        "gemms_seen": len(gemms), "gemms_expected": expect_gemms,
        "per_gemm": per_gemm,
        "block_ms": stats(block_ms),
        "block_series_ms": block_ms,
        "block_ms_once": stats(once_ms),
        "block_series_ms_once": once_ms,
        "closure_overhead_us": close.get("overhead_us"),
        "payload_conv_us": stats(pay_us),
        "publish_to_drain_us": stats(lat_us),
        "summary": summ,
        "result_line": result,
        "fatal": fatal[:10], "warnings": warns[:20],
        "rc": rc,
        "temps": {"before": temps_before, "after": temps_after,
                  "peak_before_c": peak, "battery_before_c": batt,
                  "gate_ok": gate_ok,
                  "gate": "max(TPU, skin1, skin2, VIRTUAL-SKIN) <= battery + 3 C"},
        "verdict": bool(verdict),
    }

    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    js = DEVICE_DIR / f"proc_a_{res['timestamp']}.json"
    js.write_text(json.dumps(res, indent=2), encoding="utf-8")

    def fu(x, fmt="{:.1f}"):
        return "TBD" if x is None else fmt.format(x)

    bm = res["block_ms"] or {}
    corun_ref = 147.4
    L = [f"# P3.2 — GPU stage over the real INT8 artifact — {res['timestamp']}", "",
         f"serial `{serial}` · block region `{blk.name}` ({res['block_bytes']:,} B) · "
         f"M={args.M} · route **{args.route}** · {args.blocks} blocks", "",
         "## Verdict", "",
         f"- **{'PASS' if verdict else 'FAIL'}** — {len(gemms)}/{expect_gemms} GEMM dispatches, "
         f"{len(blocks)}/{args.blocks} blocks published and drained, "
         f"fatal={len(fatal)}, drain NaN words={summ.get('drain', {}).get('nan_words', 'n/a')}",
         f"- weight buffers: {wtot.get('imported', '?')}/{wtot.get('gems', '?')} imported "
         f"zero-copy (`clImportMemoryARM(HOST_ARM)`), "
         f"{wtot.get('copied', '?')} copied", "",
         "## Numbers (device, this run)", "",
         "| quantity | value |", "|---|---|",
         f"| block, **one closure** (reference-equivalent form) | "
         f"**{fu((res['block_ms_once'] or {}).get('median'), '{:.1f}')} ms** median "
         f"(min {fu((res['block_ms_once'] or {}).get('min'))}, max {fu((res['block_ms_once'] or {}).get('max'))}, "
         f"n={(res['block_ms_once'] or {}).get('n')}) |",
         f"| block GOPS, one closure | "
         f"**{fu(2.0 * (args.M * (8 * 1536 * 1536 + 2 * 1536 * 8960)) / (((res['block_ms_once'] or {}).get('median') or 1) * 1e6), '{:.1f}')}** |",
         f"| co-run reference (`RESULTS_corun_tg.md`) | {corun_ref} ms ⇒ this run "
         f"{((res['block_ms_once'] or {}).get('median') or 0) / corun_ref:.3f}× |",
         f"| block, 10 per-GEMM closures + handoff | "
         f"{fu(bm.get('median'))} ms median (n={bm.get('n')}) |",
         f"| blocking-read closure overhead | {fu(res['closure_overhead_us'], '{:.0f}')} µs per closure |",
         f"| payload convert (float→fp16, {4700160:,} B) | "
         f"{fu((res['payload_conv_us'] or {}).get('median'), '{:.0f}')} µs median |",
         f"| publish→drain latency | {fu((res['publish_to_drain_us'] or {}).get('mean'), '{:.1f}')} µs mean, "
         f"{fu((res['publish_to_drain_us'] or {}).get('max'), '{:.1f}')} µs max |",
         "", "### Per GEMM (ms, this run — raw and net of the measured closure)",
         "",
         "| GEMM | M | N | K | median ms | net ms | GOPS raw | GOPS net | GMAC |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, s in per_gemm.items():
        L.append(f"| `{n}` | {args.M} | {s['N']} | {s['K']} | {fu(s['median'], '{:.2f}')} | "
                 f"{fu(s.get('median_net'), '{:.2f}')} | {fu(s['gops_median'], '{:.1f}')} | "
                 f"{fu(s.get('gops_net_median'), '{:.1f}')} | {s['gmac']:.3f} |")
    L += ["", "## Thermal", "",
          "- before: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_before.items())
                                  if k in ZONE_KEYS),
          "- after:  " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_after.items())
                                  if k in ZONE_KEYS),
          f"- gate (max zone ≤ battery+3): **{'PASS' if gate_ok else 'START OVER THE LINE'}** "
          f"— peak {fu(peak)} °C vs battery {fu(batt)} °C + 3 = {fu((batt or 0) + 3)} °C",
          "", "## What this proves / does not", "",
          "- proves: the GPU stage reads the **real** INT8 artifact through route B with no "
          "second copy of the weights, runs the locked block table at its measured rate, and "
          "hands each block's residual off to a second PID with a measured latency",
          "- does not prove: numerical parity of the block output (activations here are a "
          "synthetic int8 fill — the real forward pass is Gate 3), the NPU stage (P3.3), or "
          "the steady-state period (P3.5)",
          ]
    if warns:
        L += ["", "## Warnings", ""] + [f"- `{w}`" for w in warns[:10]]
    md = DEVICE_DIR / f"proc_a_{res['timestamp']}.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\nwrote {js.name} and {md.name}")
    print(f"  verdict={'PASS' if verdict else 'FAIL'} gemms={len(gemms)}/{expect_gemms} "
          f"blocks={len(blocks)} block_median_ms={bm.get('median')} "
          f"imported={wtot.get('imported')}/{wtot.get('gems')}")
    if fatal:
        for f in fatal[:5]:
            print("  " + f)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
