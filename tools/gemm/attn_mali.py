#!/usr/bin/env python3
"""Mid-block VAE attention GEMM on Mali — the GPU half of the last `T_V` term.

`T_V` has been a lower bound because the decoder's 6120-token x 384-ch mid-block
attention (32.4 GMAC, 0.63 % of the decode) had no measured rate on either engine.
The NPU half is `npu_block attn` (docs/device/attn_tv_*.md). This is the GPU half:
`gemm_bench gemm <M> <N> <K>` — one dispatch of an arbitrary INT8 GEMM at the
block-table variant (`tiled_64x64x32_t8x4`), the same variant that produced the
877 GOPS block number.

Unit of measurement = one token tile at M = 384, so 16 tiles cover 6120 tokens and
the two branches are like-for-like. Shapes:

    score  [384, 6120, 384]  -> QK^T, and by symmetry AV
    proj   [384,  384, 384]  -> q, k, v, o projections
    wide   [6120, 6120, 384] -> the same GEMM as ONE dispatch (Mali can; the NPU
                                cannot, its graph ceiling is 384 tokens). Run last:
                                it allocates two 150 MB buffers.

    python tools/gemm/attn_mali.py [--skip-wide]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "probe"))
from probe_host import ADB, adb, pick_serial, read_zones, run  # noqa: E402

GEMM_DIR = REPO / "tools" / "android" / "gemm"
BIN = GEMM_DIR / "build" / "gemm_bench"
CL = GEMM_DIR / "kernels" / "gemm8.cl"
REMOTE_BIN = "/data/local/tmp/gemm_bench"
REMOTE_CL = "/data/local/tmp/gemm8.cl"
DEVICE_DIR = REPO / "docs" / "device"
ZONE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")
VARIANT = "tiled_64x64x32_t8x4"

# N (and M) must be a multiple of the kernel's 64-wide output tile: at N=6120 the
# tail tile writes past C and the blocking read returns CL_EXEC_STATUS_ERROR (-14)
# with an impossible 10.5 TOPS "rate". 6120 tokens are therefore padded to 6144
# (+0.39 % MACs) and the padding is recorded, not hidden. N=384 is exact, which is
# why the `proj` shape needs no pad.
PAD_NOTE = "N padded 6120 -> 6144 for the 64-wide output tile (+0.39 % MACs)"
SHAPES = [
    ("score", 384, 6144, 384, 8),      # one token tile; x16 = the tiled pass
    ("proj",  384,  384, 384, 8),      # one token tile; x16 = the tiled pass
    ("wide", 6144, 6144, 384, 2),      # QK^T as ONE dispatch (Mali can)
    ("projw", 6144, 384, 384, 4),      # q/k/v/o as ONE dispatch each
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-wide", action="store_true")
    ap.add_argument("--cooldown", type=float, default=None)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    if not BIN.exists():
        print(f"{BIN} not built — run bash tools/android/gemm/build.sh", file=sys.stderr)
        return 2
    serial = args.serial or pick_serial()
    if not serial:
        print("no device", file=sys.stderr)
        return 2

    before = read_zones(serial)
    peak = max((v for k, v in before.items() if k in ZONE_KEYS), default=None)
    print(f"thermal: peak={peak} C, battery={before.get('battery')} C"
          + (f", gate={args.cooldown} C" if args.cooldown else ""))
    if args.cooldown is not None and peak is not None and peak > args.cooldown:
        print("TOO WARM — refusing to report a throttled run", file=sys.stderr)
        return 3

    run([str(ADB), "-s", serial, "push", str(BIN), REMOTE_BIN], timeout=180)
    run([str(ADB), "-s", serial, "push", str(CL), REMOTE_CL], timeout=180)
    adb(serial, f"shell chmod 755 {REMOTE_BIN}")

    runs = []
    for name, M, N, K, iters in SHAPES:
        if args.skip_wide and name == "wide":
            continue
        cmd = f"shell {REMOTE_BIN} gemm {REMOTE_CL} {VARIANT} {iters} {M} {N} {K}"
        print(f"running: {cmd}")
        rc, out = adb(serial, cmd, timeout=args.timeout)
        print(out.strip())
        rec = {"name": name, "M": M, "N": N, "K": K, "iters": iters, "rc": rc,
               "raw": out.splitlines()}
        for line in out.splitlines():
            if line.startswith("ATTN_GEMM|"):
                d = dict(kv.split("=", 1) for kv in line.split("|")[1:] if "=" in kv)
                rec["ms"] = float(d["ms"])
                rec["gops"] = float(d["gops"])
                # tiles: 16 token tiles make up 6120 tokens; a 6144-row shape IS
                # the whole pass in one dispatch. The binary's own per_pass_16t
                # field assumes 16 and is wrong for the wide shapes, so it is not
                # used here.
                rec["tiles"] = 16 if M <= 384 else 1
                rec["per_pass_s"] = rec["ms"] * rec["tiles"] / 1000.0
            elif line.startswith("ENQ_FAIL") or line.startswith("ATTN|"):
                rec.setdefault("errors", []).append(line)
        runs.append(rec)

    after = read_zones(serial)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_rec = {"serial": serial, "timestamp": ts, "variant": VARIANT, "shapes": SHAPES,
               "pad_note": PAD_NOTE,
               "temps": {"before": before, "after": after}, "runs": runs}
    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    p = DEVICE_DIR / f"attn_mali_{ts}.json"
    p.write_text(json.dumps(out_rec, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    for r in runs:
        if "ms" in r:
            print(f"  {r['name']}: {r['ms']:.3f} ms  ({r['gops']:.1f} GOPS) "
                  f"({r['tiles']} dispatch(es)) per pass {r['per_pass_s']:.4f} s")
        else:
            print(f"  {r['name']}: NO RESULT rc={r['rc']} {r.get('errors')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
