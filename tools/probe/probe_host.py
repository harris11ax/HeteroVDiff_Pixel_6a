#!/usr/bin/env python3
"""Pixel 6a device probe — host side (Phase 2, Gate 0).

Runs the on-device `probe` binary plus a batch of `adb shell` facts, and writes
everything to docs/device/probe_<serial>_<ts>.{json,md}.

    python tools/probe/probe_host.py --build          # build + run everything
    python tools/probe/probe_host.py --preflight      # what is (not) available
    python tools/probe/probe_host.py --cooldown 30    # block until <= 30 C
    python tools/probe/probe_host.py --only ahb

Nothing here needs root or an APK. The probe is a plain PIE executable in
/data/local/tmp.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SDK = Path(os.environ.get("ANDROID_SDK_ROOT", r"C:/Users/harri/AppData/Local/Android/Sdk"))
ADB = SDK / "platform-tools" / "adb.exe"
PROBE_DIR = REPO / "tools" / "android" / "probe"
PROBE_BIN = PROBE_DIR / "build" / "probe"
DEVICE_DIR = REPO / "docs" / "device"
REMOTE = "/data/local/tmp/hv_probe"


def run(cmd: list[str] | str, timeout: int = 120, shell: bool = False) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=shell)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "<timeout>"
    except FileNotFoundError as e:
        return 127, f"<missing: {e}>"


def adb(serial: str | None, args: str, timeout: int = 120) -> tuple[int, str]:
    base = [str(ADB)] + (["-s", serial] if serial else [])
    return run(base + args.split(), timeout=timeout)


def pick_serial() -> str | None:
    rc, out = run([str(ADB), "devices"])
    if rc != 0:
        return None
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            return parts[0]
    return None


def preflight() -> int:
    print(f"adb            : {ADB}  ({'found' if ADB.exists() else 'MISSING'})")
    print(f"ndk            : {SDK / 'ndk'}")
    print(f"probe binary   : {PROBE_BIN}  ({'built' if PROBE_BIN.exists() else 'not built'})")
    serial = pick_serial()
    print(f"device         : {serial or 'NONE ATTACHED'}")
    if not serial:
        print("\nAttach the Pixel 6a over USB, enable USB debugging, then:")
        print("  adb devices            # must list it as 'device', not 'unauthorized'")
        return 1
    for key in ("ro.product.model", "ro.board.platform", "ro.soc.model", "ro.build.version.release"):
        rc, out = adb(serial, f"shell getprop {key}")
        print(f"{key:30s}: {out.strip()}")
    return 0


# ---------------------------------------------------------------- shell facts
SHELL_FACTS = [
    ("props", "shell getprop"),
    ("meminfo", "shell cat /proc/meminfo"),
    ("dumpsys_meminfo", "shell dumpsys meminfo"),
    ("cpuinfo", "shell cat /proc/cpuinfo"),
    ("thermal_zones", "shell sh -c 'for z in /sys/class/thermal/thermal_zone*; do echo $z $(cat $z/type) $(cat $z/temp); done'"),
    ("thermalservice", "shell dumpsys thermalservice"),
    ("dmabuf_heaps", "shell ls -l /dev/dma_heap /dev/dma-buf 2>/dev/null"),
    ("opencl_libs", "shell sh -c 'ls -l /vendor/lib64/libOpenCL* /vendor/lib/libOpenCL* /system/lib64/libOpenCL* 2>/dev/null'"),
    ("nnapi_libs", "shell sh -c 'ls -l /vendor/lib64/libneuralnetworks* /vendor/lib64/hw/neuralnetworks* 2>/dev/null'"),
    ("tpu_props", "shell sh -c 'getprop | grep -iE \"tpu|edgetpu|nnapi|darwinn|gxp|dsp\"'"),
    ("gpu_info", "shell sh -c 'cat /sys/class/kgsl/kgsl-3d0/gpu_model 2>/dev/null; getprop ro.hardware.egl'"),
    ("thermal_budget", "shell sh -c 'getprop | grep -iE \"thermal|powerhint|perf\"'"),
]


def collect_shell(serial: str, out: dict, verbose: bool = True) -> None:
    for name, args in SHELL_FACTS:
        rc, txt = adb(serial, args, timeout=60)
        out[name] = {"rc": rc, "text": txt}
        if verbose:
            print(f"  shell/{name:<18} rc={rc} ({len(txt)} bytes)")


# ------------------------------------------------------------------- parsing
KV = re.compile(r"^([A-Za-z0-9_\[\]\.]+)=(.*)$")


def parse_probe(stdout: str) -> dict:
    kv: dict[str, str] = {}
    in_block = False
    for line in stdout.splitlines():
        if line.strip() == "==JSON_BEGIN==":
            in_block = True
            continue
        if line.strip() == "==JSON_END==":
            in_block = False
            continue
        if not in_block:
            continue
        m = KV.match(line.strip())
        if m:
            kv[m.group(1)] = m.group(2)
        elif line.strip():
            kv.setdefault("_log", "")
            kv["_log"] += line + "\n"
    return kv


def run_probe(serial: str, mode: str, timeout: int = 900) -> dict:
    if not PROBE_BIN.exists():
        raise SystemExit("probe binary missing — run with --build")
    rc, txt = run([str(ADB), "-s", serial, "push", str(PROBE_BIN), REMOTE])
    if rc != 0:
        raise SystemExit(f"adb push failed: {txt}")
    adb(serial, f"shell chmod 755 {REMOTE}")
    print(f"  running probe {mode} ... (may take a few minutes; AHB sweep can wedge gralloc)")
    rc, txt = adb(serial, f"shell {REMOTE} {mode}", timeout=timeout)
    kv = parse_probe(txt)
    kv["_rc"] = str(rc)
    return kv


# ------------------------------------------------------------------ cooldown
TEMP_RE = re.compile(r"Temperature\{mValue=([-0-9.]+), mType=(-?\d+), mName=([A-Za-z0-9_\-]+)")


def read_zones(serial: str) -> dict[str, float]:
    """Named thermal zones from dumpsys thermalservice.

    /sys/class/thermal/thermal_zone*/temp is Permission denied for the shell user
    on Android 17 (verified), so dumpsys is the only unprivileged source. The
    'Current temperatures from HAL' block is the live one; earlier blocks are
    cached values, so the LAST occurrence of each name wins.
    """
    rc, txt = adb(serial, "shell dumpsys thermalservice", timeout=60)
    if rc != 0:
        return {}
    zones: dict[str, float] = {}
    for m in TEMP_RE.finditer(txt):
        try:
            zones[m.group(3)] = float(m.group(1))
        except ValueError:
            continue
    return zones


def read_max_temp(serial: str, zones_wanted: list[str] | None = None) -> float | None:
    zones = read_zones(serial)
    if not zones:
        return None
    if zones_wanted:
        sel = [v for k, v in zones.items() if k in zones_wanted]
    else:
        sel = list(zones.values())
    return max(sel) if sel else None


def cooldown(serial: str, target: float, poll: int = 30, max_wait: int = 3600) -> int:
    """Spec item 7: no timed run until the SoC is <= target C."""
    t0 = time.time()
    while True:
        z = read_zones(serial)
        # TPU + GPU + skin are the zones that matter; fall back to all of them
        keep = {k: v for k, v in z.items() if k in ("TPU", "skin_therm1", "skin_therm2",
                                                    "battery", "VIRTUAL-SKIN")} or z
        t = max(keep.values()) if keep else None
        el = int(time.time() - t0)
        detail = " ".join(f"{k}={v:.1f}" for k, v in sorted(keep.items()))
        print(f"[{el:5d}s] max = {t if t is None else f'{t:.1f} C'}   ({detail})")
        if t is not None and t <= target:
            print(f"READY at {t:.1f} C after {el}s")
            return 0
        if el > max_wait:
            print("TIMEOUT waiting for cooldown")
            return 1
        time.sleep(poll)


# --------------------------------------------------------------------- main
def summarize(kv: dict) -> list[str]:
    L = []
    def g(k, d="?"):
        return kv.get(k, d)
    L.append(f"- OpenCL lib: `{g('CL_LIB')}`")
    L.append(f"- OpenCL platforms: **{g('CL_PLATFORMS')}**")
    for k in sorted(kv):
        if re.match(r"CL_DEV\[0\]\.(NAME|VERSION|MAX_MEM_ALLOC_MiB|GLOBAL_MEM_MiB|HAS_ARM_DOT|HAS_ARM_DOT_INT8|HAS_ARM_IMPORT_MEMORY)$", k):
            L.append(f"- `{k}` = `{kv[k]}`")
    for tag in ("ARMDOT", "ARMDOT_INT8", "SCALAR"):
        line = next((kv[k] for k in kv if k.startswith(f"GEMM_{tag}") and k.endswith("GOPS")), None)
        if line:
            L.append(f"- GEMM {tag}: GOPS={line}")
    L.append(f"- AHB max alloc that succeeded: **{g('AHB_MAX_OK_MiB')} MiB**")
    L.append(f"- AHB->NNAPI memory rc: `{g('ZEROCOPY_NNAPI_MEMORY_rc')}` (0 = OK)")
    L.append(f"- AHB->OpenCL import: `{g('ZEROCOPY_CL_IMPORT_dev0_mem')}` err=`{g('ZEROCOPY_CL_IMPORT_dev0_err')}`")
    L.append(f"- NNAPI devices: **{g('NNAPI_DEVICE_COUNT_rc')}** / count={g('NNAPI_DEVICE_COUNT_rc')}")
    for k in sorted(kv):
        if re.match(r"NNAPI_DEV\[\d+\]\.(NAME|TYPE|fl)$|NNAPI_FC.*", k):
            L.append(f"- `{k}` = `{kv[k]}`")
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="rebuild the NDK probe first")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--cooldown", type=float, default=None, metavar="C")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--only", default="all", choices=["all", "opencl", "ahb", "nnapi"])
    ap.add_argument("--no-shell", action="store_true")
    args = ap.parse_args()

    if not ADB.exists():
        print(f"adb not found at {ADB}", file=sys.stderr)
        return 2
    if args.preflight:
        return preflight()

    if args.build:
        print("building probe ...")
        rc, txt = run(["bash", str(PROBE_DIR / "build.sh")], timeout=900)
        print(txt[-2000:] if rc else "build ok")

    serial = args.serial or pick_serial()
    if not serial:
        print("No device. Run --preflight for instructions.", file=sys.stderr)
        return 2

    if args.cooldown is not None:
        return cooldown(serial, float(args.cooldown))

    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    record: dict = {"serial": serial, "timestamp": ts, "probe": {}, "temp_before_c": read_max_temp(serial)}

    if not args.no_shell:
        print("collecting shell facts ...")
        collect_shell(serial, record)

    print("running on-device probe ...")
    record["probe"] = run_probe(serial, args.only)
    record["temp_after_c"] = read_max_temp(serial)

    js = DEVICE_DIR / f"probe_{ts}.json"
    js.write_text(json.dumps(record, indent=2), encoding="utf-8")

    md = DEVICE_DIR / f"probe_{ts}.md"
    lines = [f"# Pixel 6a probe — {ts}", "", f"serial `{serial}`", "",
             f"thermal before/after: {record['temp_before_c']} / {record['temp_after_c']} C", "",
             "## Findings", ""] + summarize(record["probe"])
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {js}\nwrote {md}")
    print("\n".join(summarize(record["probe"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
