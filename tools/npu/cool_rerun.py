#!/usr/bin/env python3
"""Gate-and-run: wait for the idle thermal floor, then take the cool T_N re-run.

The locked rule is `max(TPU, skin_therm1, skin_therm2, VIRTUAL-SKIN) <= battery + 3 C`
at the moment a *timed* run starts. `npu_host.py --cooldown` is a one-shot
threshold check, not a wait, so this wrapper does the waiting and then hands the
harness the same threshold it just passed — a double gate on the same sample.

Everything is timestamped to a log so a run that is still cooling can be told
apart from one that is hung, and the run is designed to be safe to repeat.

    python tools/npu/cool_rerun.py --max-wait 3600
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "probe"))
from probe_host import pick_serial, read_zones  # noqa: E402

GATE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "VIRTUAL-SKIN")
LOG = REPO / "docs" / "device" / ".cool_rerun.log"


def stamp(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sample(serial: str) -> tuple[float, float, dict[str, float]]:
    z = read_zones(serial)
    peak = max((v for k, v in z.items() if k in GATE_KEYS), default=float("nan"))
    batt = z.get("battery", float("nan"))
    return peak, batt, z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-wait", type=float, default=3600, help="give up cooling after this many s")
    ap.add_argument("--poll", type=float, default=45, help="s between thermal samples while cooling")
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--iters", type=int, default=None,
                    help="override the harness iteration count (default: 2 for single, 3 otherwise)")
    ap.add_argument("--modes", default="single",
                    help="harness modes to run once the gate passes (single|downsep|down2|...)")
    ap.add_argument("--M", type=int, default=384)
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    serial = pick_serial()
    if not serial:
        stamp("FATAL no device")
        return 2
    stamp(f"START cool-rerun serial={serial} max_wait={args.max_wait:.0f}s poll={args.poll:.0f}s")

    t0 = time.time()
    while True:
        peak, batt, z = sample(serial)
        gate = batt + 3.0
        stamp("zones " + " ".join(f"{k}={z[k]:.1f}" for k in GATE_KEYS if k in z)
              + f" | peak={peak:.1f} gate={gate:.1f} ({'PASS' if peak <= gate else 'waiting'})")
        if peak <= gate:
            break
        if time.time() - t0 > args.max_wait:
            stamp(f"GAVE UP after {time.time() - t0:.0f}s: peak={peak:.1f} > gate={gate:.1f}; no timed run")
            return 3
        time.sleep(args.poll)

    time.sleep(10)  # settle, then confirm on a fresh sample
    peak, batt, z = sample(serial)
    gate = batt + 3.0
    stamp(f"CONFIRM peak={peak:.1f} gate={gate:.1f}")
    if peak > gate:
        stamp("FATAL gate failed on confirm sample")
        return 3

    cmd = [sys.executable, str(REPO / "tools" / "npu" / "npu_host.py"),
           "--modes", args.modes, "--pc", "0", "--tiles", str(args.tiles),
           "--M", str(args.M), "--cooldown", f"{gate:.1f}", "--serial", serial]
    if args.iters is not None:
        cmd += ["--iters", str(args.iters)]
    stamp("RUN " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(REPO))
    stamp(f"HARNESS EXIT rc={rc}")
    stamp("=== RUNNER EXIT ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())
