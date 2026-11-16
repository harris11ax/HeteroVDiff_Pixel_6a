#!/usr/bin/env python3
"""Sustained-NPU run — does the Edge TPU hold its block rate for a whole chunk?

`T_N` was measured as *short* bursts: 9 compiles held, then 100 rounds of
9 GEMMs (one round = one block's compilable GEMM set). The cut then assumes
that rate survives **20 back-to-back blocks + a 10.6 s decode** on the NPU side,
i.e. ~23.6 s of continuous Tensor-G1 work inside a 23.61 s period. A rate that
decays at second 15 of every chunk would silently inflate the period and no
short capture would show it.

This drives the existing `npu_block corun` loop long enough to cover the NPU
stage's whole continuous length and reports the drift:

    python tools/npu/sustain_npu.py --rounds 60          # 60 x 0.39 s ~= 23.4 s
    python tools/npu/sustain_npu.py --cooldown 24.9      # gate the start

Verdict threshold is 2 % on the last-decile / first-decile mean, chosen because
the per-GEMM spread under co-run was measured at +-4 % (RESULTS_corun_tg.md);
anything inside that is noise, anything above it is throttling or DVFS.

Nothing is modelled. If the run dies the artifact says so and the rates stay null.
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

NPU_DIR = REPO / "tools" / "android" / "npublock"
NPU_BIN = NPU_DIR / "build" / "npu_block"
DEVICE_DIR = REPO / "docs" / "device"
ZONE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")
GATE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "VIRTUAL-SKIN")
DRIFT_LIMIT = 0.02          # 2 %: inside the measured +-4 % per-GEMM co-run noise


def kv(line: str) -> dict:
    d = {}
    for part in line.split("|")[1:]:
        k, sep, v = part.partition("=")
        if sep:
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d


def stats(v: list[float]) -> dict | None:
    if not v:
        return None
    return {"n": len(v), "min": min(v), "median": st.median(v),
            "mean": sum(v) / len(v), "max": max(v),
            "stdev": st.pstdev(v) if len(v) > 1 else 0.0}


def slope_ms_per_round(rounds: list[dict]) -> float | None:
    """Least-squares slope of round duration against round index (ms/round)."""
    n = len(rounds)
    if n < 3:
        return None
    xs = [r["r"] for r in rounds]
    ys = [r["ms"] for r in rounds]
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=60, help="rounds of 9 GEMMs (= blocks)")
    ap.add_argument("--tiles", type=int, default=4, help="executes per GEMM per round (4 = preset A)")
    ap.add_argument("--M", type=int, default=384, help="tokens per graph (384 = NNAPI ceiling)")
    ap.add_argument("--cooldown", type=float, default=None,
                    help="refuse to start unless max(TPU,skin,VIRTUAL-SKIN) <= ambient+this")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--serial", default=None)
    args = ap.parse_args()

    if args.M > 384:
        print("M above the measured per-graph ceiling (384) — refusing", file=sys.stderr)
        return 2
    if not ADB.exists():
        print(f"adb missing at {ADB}", file=sys.stderr)
        return 2
    if not NPU_BIN.exists():
        print(f"{NPU_BIN} not built — run tools/android/npublock/build.sh", file=sys.stderr)
        return 2

    serial = args.serial or pick_serial()
    if not serial:
        print("no device", file=sys.stderr)
        return 2

    temps_before = read_zones(serial)
    peak_before = max((v for k, v in temps_before.items() if k in GATE_KEYS), default=None)
    batt_before = temps_before.get("battery", float("nan"))
    table_grade = (peak_before is not None and peak_before <= batt_before + 3.0)
    if args.cooldown is not None:
        gate = batt_before + args.cooldown
        print(f"thermal gate: peak={peak_before} C vs battery+{args.cooldown} = {gate:.1f} C")
        if peak_before is None or peak_before > gate:
            print("TOO WARM — refusing a throttled run (use tools/npu/cool_rerun.py)",
                  file=sys.stderr)
            return 3

    rc, txt = run([str(ADB), "-s", serial, "push", str(NPU_BIN), "/data/local/tmp/npu_block"],
                  timeout=180)
    if rc != 0:
        print(f"push failed: {txt}", file=sys.stderr)
        return 2
    adb(serial, "shell chmod 755 /data/local/tmp/npu_block")
    adb(serial, "logcat -c")

    cmd = f"shell /data/local/tmp/npu_block corun {args.tiles} {args.rounds} {args.M} 0"
    print(f"running: {cmd}  ({args.rounds} rounds ~= {args.rounds * 0.39:.1f} s of TPU work)")
    rc, out = adb(serial, cmd, timeout=args.timeout)
    temps_after = read_zones(serial)

    rounds = [kv(l) for l in out.splitlines() if l.strip().startswith("CORUN_ROUND|")]
    executes = [kv(l) for l in out.splitlines() if l.strip().startswith("TS|npu|")]
    per_gemm_lines = [l.strip() for l in out.splitlines() if l.strip().startswith("OP|")]
    fatal = [l.strip() for l in out.splitlines() if l.strip().startswith("FATAL")]
    summary = [kv(l) for l in out.splitlines() if l.strip().startswith("SUMMARY|")]

    # group executes by GEMM name in run order, so a per-GEMM drift is visible
    per_gemm: dict[str, list[float]] = {}
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("TS|npu|"):
            per_gemm.setdefault(s.split("|")[2], []).append(kv(s)["ms"])

    r_ms = [r["ms"] for r in rounds]
    n = len(r_ms)
    third = max(1, n // 5)
    head = r_ms[:third]
    tail = r_ms[-third:]
    head_mean = sum(head) / len(head) if head else None
    tail_mean = sum(tail) / len(tail) if tail else None
    drift = (tail_mean / head_mean - 1.0) if head_mean and tail_mean else None
    slope = slope_ms_per_round(rounds)
    sustained = None if drift is None else abs(drift) <= DRIFT_LIMIT

    gemm_drift = {}
    for name, v in per_gemm.items():
        k = max(1, len(v) // 5)
        h, t = v[:k], v[-k:]
        if h and t:
            gemm_drift[name] = {"n": len(v), "first5pct_mean_ms": sum(h) / len(h),
                                "last5pct_mean_ms": sum(t) / len(t),
                                "drift_pct": (sum(t) / len(t)) / (sum(h) / len(h)) * 100 - 100}

    res = {
        "serial": serial,
        "timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "mode": "corun", "rounds": args.rounds, "tiles": args.tiles, "M": args.M, "rc": rc,
        "temps": {"before": temps_before, "after": temps_after},
        "gate": {
            "peak_before_c": peak_before,
            "battery_before_c": batt_before,
            "rule": "max(TPU, skin_therm1, skin_therm2, VIRTUAL-SKIN) <= battery + 3 C",
            "table_grade": table_grade,
            "forced_cooldown_arg": args.cooldown},
        "rounds_ms": r_ms,
        "round_stats": stats(r_ms),
        "first_20pct_mean_ms": head_mean,
        "last_20pct_mean_ms": tail_mean,
        "drift_pct_last_over_first": None if drift is None else drift * 100,
        "drift_slope_ms_per_round": slope,
        "drift_within_noise": sustained,
        "npu_busy_s": sum(r_ms) / 1000.0 if r_ms else None,
        "per_block_s_first": None if head_mean is None else head_mean / 1000.0,
        "per_block_s_last": None if tail_mean is None else tail_mean / 1000.0,
        "per_gemm_drift": gemm_drift,
        "summary": summary,
        "fatal": fatal,
    }

    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    ts = res["timestamp"]
    js = DEVICE_DIR / f"sustain_npu_{ts}.json"
    js.write_text(json.dumps(res, indent=2), encoding="utf-8")

    def f(x, fmt="{:.3f}"):
        return "TBD" if x is None else fmt.format(x)

    L = [f"# Sustained NPU run — {ts}", "",
         f"serial `{serial}` · `corun` M={args.M} tiles={args.tiles} rounds={args.rounds} · "
         f"rc={rc} · {f(res['npu_busy_s'], '{:.1f}')} s of continuous TPU work", "",
         "## Thermal", "",
         "- before: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_before.items()) if k in ZONE_KEYS),
         "- after:  " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_after.items()) if k in ZONE_KEYS)]
    if res["gate"]:
        g = res["gate"]
        L.append(f"- gate: peak {f(g['peak_before_c'], '{:.1f}')} C vs "
                 f"battery+3 = {g['battery_before_c'] + 3.0:.1f} C -> "
                 + ("table-grade start" if g["table_grade"] else
                    "**NOT table-grade** (started above the line; relative drift still valid, "
                    "absolute rate is not)"))
    L += ["", "## Round duration (one round = one block's 9 compilable GEMMs)", "",
          "| | ms |", "|---|---|",
          f"| first 20 % (rounds 0-{third - 1}) | {f(head_mean)} |",
          f"| last 20 % (rounds {n - third}-{n - 1}) | {f(tail_mean)} |",
          f"| all rounds mean | {f(res['round_stats'] and res['round_stats']['mean'])} |",
          f"| all rounds stdev | {f(res['round_stats'] and res['round_stats']['stdev'])} |",
          f"| min / median / max | {f(res['round_stats'] and res['round_stats']['min'])} / "
          f"{f(res['round_stats'] and res['round_stats']['median'])} / "
          f"{f(res['round_stats'] and res['round_stats']['max'])} |", "",
          f"**Drift: {f(res['drift_pct_last_over_first'], '{:+.2f}')} %** "
          f"(slope {f(slope, '{:+.4f}')} ms/round) — "
          + ("TBD: the run did not produce rounds (see the FATAL lines below)."
             if sustained is None else
             "inside the +-4 % co-run noise, the rate is sustained."
             if sustained else "OUTSIDE the noise floor: the rate decays, so a chunk-period "
             "prediction from a short burst would be optimistic."), ""]
    if gemm_drift:
        L += ["## Per-GEMM drift (first 5 % vs last 5 % of that GEMM's executes)", "",
              "| GEMM | n | first ms | last ms | drift |", "|---|---|---|---|---|"]
        for name, d in sorted(gemm_drift.items(), key=lambda kv_: -abs(kv_[1]["drift_pct"])):
            L.append(f"| `{name}` | {d['n']} | {f(d['first5pct_mean_ms'])} | "
                     f"{f(d['last5pct_mean_ms'])} | {d['drift_pct']:+.1f} % |")
        L.append("")
    if summary:
        s = summary[0]
        L.append(f"- harness summary: {f(s.get('per_block_s'), '{:.4f}')} s/block, "
                 f"{f(s.get('gops'), '{:.1f}')} GOPS, compile {f(s.get('compile_ms'), '{:.0f}')} ms")
    for fl in fatal:
        L.append(f"- `{fl}`")
    L.append("")
    md = DEVICE_DIR / f"sustain_npu_{ts}.md"
    md.write_text("\n".join(L), encoding="utf-8")

    print("\n".join(L))
    print(f"\nwrote {js}\nwrote {md}")
    return 0 if rc == 0 and not fatal else 1


if __name__ == "__main__":
    sys.exit(main())
