#!/usr/bin/env python3
"""T_V — VAE decode rate on the Tensor Edge TPU (NNAPI `CONV_2D` temporal taps).

The Wan decoder is a stack of causal 3-D convs; NNAPI has no `CONV_3D` (Gate 0).
Branch measured here: decompose every `kT=3` causal conv into `kT` shifted 2-D
`CONV_2D` ops over the (h, w) plane of one frame, accumulated into the output
frame. The unit of measurement is therefore ONE `CONV_2D` at the layer's real
shape — the same construction as the `T_N` per-GEMM table.

Nothing is modelled. Every number in the report is a wall-clock around a
synchronous `execute` on the phone. A shape that will not compile or run stays
`null` and the ladder that failed is recorded.

    python tools/npu/conv_vae_host.py --plan
    python tools/npu/conv_vae_host.py --run --only class3 --tiles 3 --iters 2
    python tools/npu/conv_vae_host.py --report
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "probe"))
from probe_host import adb, pick_serial, read_zones  # noqa: E402

NPU_DIR = REPO / "tools" / "android" / "npublock"
BIN = NPU_DIR / "build" / "npu_block"
REMOTE_BIN = "/data/local/tmp/npu_block"
DEVICE_DIR = REPO / "docs" / "device"
OPS_JSON = REPO / "configs" / "vae_decode_ops.json"

GATE_ZONES = ("TPU", "skin_therm1", "skin_therm2", "VIRTUAL-SKIN", "battery")
# measured bracket (see ladder()): batched CONV_2D compiles at 0.78 MB/frame and
# FAILS after 31-33 s at 1.4 MB/frame, so the usable boundary is between them.
MAX_BATCH_FRAME_MB = 1.0


def kv_lines(stdout: str) -> list[dict]:
    out = []
    for line in stdout.splitlines():
        if "|" not in line:
            continue
        parts = line.strip().split("|")
        rec = {"tag": parts[0]}
        for p in parts[1:]:
            if "=" in p:
                k, _, v = p.partition("=")
                rec[k] = v
        out.append(rec)
    return out


def first(recs: list[dict], tag: str) -> dict | None:
    for r in recs:
        if r["tag"] == tag:
            return r
    return None


def gate_ok(zones: dict) -> tuple[bool | None, float | None, float | None]:
    if not zones or "battery" not in zones:
        return None, None, None
    hot = max(v for k, v in zones.items() if k in GATE_ZONES and k != "battery")
    gate = zones["battery"] + 3.0
    return hot <= gate, hot, gate


# --------------------------------------------------------------------- plan
def load_classes() -> list[dict]:
    d = json.loads(OPS_JSON.read_text(encoding="utf-8"))
    cls = []
    for name, v in d["classes"].items():
        fhw = v.get("fhw")
        if not fhw or v["kind"] != "conv":
            continue
        f, h, w = fhw
        k = v["k"]
        k = k if isinstance(k, list) else [1, k, k]
        if len(k) == 2:                 # 4-D checkpoint weight [Co, Ci, kH, kW]
            k = [1, k[0], k[1]]
        kt, kh, kw = k[0], k[1], k[2]
        if kh != kw:
            raise SystemExit(f"non-square spatial kernel in {name}")
        cls.append({
            "name": name, "f": f, "h": h, "w": w,
            "cin": v["cin"], "cout": v["cout"], "ktaps": kt, "k": kh,
            "ops": v["ops"], "macs_per_op": v["macs"],
            "total_macs": v["ops"] * v["macs"],
        })
    cls.sort(key=lambda c: -c["total_macs"])
    return cls


# ------------------------------------------------------------------ attempts
def ladder(c: dict) -> list[dict]:
    """Real decompositions only: frames are independent for a 2-D tap, so an
    N-split is exact. A spatial split perturbs the seam rows (SAME padding is
    computed per tile) and is marked `seam=True`.

    MEASURED 2026-11-15: the Edge TPU delegate compiles a batched CONV_2D when
    the per-frame input is small (N=3, 34x60x384 = 0.78 MB/frame: 141.6 ms) and
    grinds 31-36 s then fails to compile when it is large (N=3, 136x240x192 =
    6.3 MB/frame; N=9 also fails), while the SAME 37.6 MB tensor compiles fine at
    N=1 (263 ms). So the batch dim is the problem, not the byte count: above
    ~2.4 MB per frame the ladder starts at N=1 instead of burning a minute on a
    compile that cannot finish.
    """
    f, h, w = c["f"], c["h"], c["w"]
    frame_mb = h * w * c["cin"] / 2**20
    batch_ok = frame_mb <= MAX_BATCH_FRAME_MB
    tries = []
    if batch_ok:
        tries.append({"split": "none", "N": f, "H": h, "W": w, "fsplit": 1,
                      "hsplit": 1, "wsplit": 1, "seam": False})
        for n in (3,):
            if n < f:
                tries.append({"split": f"N={n}", "N": n, "H": h, "W": w,
                              "fsplit": -(-f // n), "hsplit": 1, "wsplit": 1, "seam": False})
    tries.append({"split": "N=1", "N": 1, "H": h, "W": w, "fsplit": f,
                  "hsplit": 1, "wsplit": 1, "seam": False})
    for hs in (2, 4):
        tries.append({"split": f"N=1,H/{hs}", "N": 1, "H": -(-h // hs), "W": w,
                      "fsplit": f, "hsplit": hs, "wsplit": 1, "seam": True})
    for ws in (2, 4):
        tries.append({"split": f"N=1,W/{ws}", "N": 1, "H": h,
                      "W": -(-w // ws), "fsplit": f,
                      "hsplit": 1, "wsplit": ws, "seam": True})
    return tries


def run_one(serial: str, c: dict, t: dict, tiles: int, iters: int, timeout: int) -> dict:
    args = f"{tiles} {iters} {t['N']} {t['H']} {t['W']} {c['cin']} {c['cout']} {c['k']}"
    zb = read_zones(serial)
    rc, out = adb(serial, f"shell {REMOTE_BIN} conv {args}", timeout=timeout)
    za = read_zones(serial)
    recs = kv_lines(out)
    fin = first(recs, "MODEL_FINISH")
    sup = first(recs, "SUPPORT")
    comp = first(recs, "COMPILE")
    summ = first(recs, "SUMMARY")
    op = first(recs, "OP")
    ok, hot, gate = gate_ok(zb)
    res = {
        "class": c["name"], "split": t["split"], "attempt": args, "rc": rc,
        "model_finish_rc": fin.get("rc") if fin else None,
        "delegated": sup.get("conv2d_delegated") if sup else None,
        "compile_ms": float(comp["ms"]) if comp and "ms" in comp else None,
        "ms": float(op["ms"]) if op and "ms" in op else None,
        "gops": float(op["gops"]) if op and "gops" in op else None,
        "macs_measured": int(op["macs"]) if op and "macs" in op else None,
        "zones_before": zb, "zones_after": za,
        "gate_ok": ok, "hot_before": hot, "gate": gate,
        "seam": t["seam"], "fsplit": t["fsplit"],
        "hsplit": t["hsplit"], "wsplit": t["wsplit"],
        "sub_ops": t["fsplit"] * t["hsplit"] * t["wsplit"],
    }
    res["ok"] = bool(res["ms"] is not None and rc == 0)
    # GOPS host-side from the shape (never trust the on-device field: it was
    # int32 and overflowed before 2026-11-15 18:20)
    if res["ok"]:
        n, hh, ww = t["N"], t["H"], t["W"]
        res["macs_measured"] = n * hh * ww * c["cout"] * c["cin"] * c["k"] * c["k"]
        res["gops"] = round(2.0 * res["macs_measured"] / (res["ms"] / 1000.0) / 1e9, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on the class name")
    ap.add_argument("--tiles", type=int, default=3)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    classes = load_classes()
    if args.plan:
        print(f"{'class':34s} {'N':>3s} {'H':>4s} {'W':>4s} {'Cin':>4s} {'Cout':>4s} "
              f"{'k':>2s} {'taps':>4s} {'ops':>3s} {'GMAC/op':>8s} {'tot GMAC':>9s}")
        for c in classes:
            print(f"{c['name'][:34]:34s} {c['f']:3d} {c['h']:4d} {c['w']:4d} {c['cin']:4d} "
                  f"{c['cout']:4d} {c['k']:2d} {c['ktaps']:4d} {c['ops']:3d} "
                  f"{c['macs_per_op']/1e9:8.3f} {c['total_macs']/1e9:9.2f}")
        print(f"{'TOTAL':34s} {'':>3s} {'':>4s} {'':>4s} {'':>4s} {'':>4s} {'':>2s} "
              f"{'':>4s} {'':>3s} {'':>8s} {sum(c['total_macs'] for c in classes)/1e9:9.2f}")
        return 0

    serial = pick_serial()
    if not serial:
        print("NO_DEVICE")
        return 3

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else DEVICE_DIR / f"conv_sweep_{ts}.json"

    if args.run:
        sel = [c for c in classes if not args.only or args.only in c["name"]]
        if not sel:
            print(f"no class matches --only {args.only}")
            return 2
        adb(serial, f"push {BIN} {REMOTE_BIN}", timeout=180)
        adb(serial, f"shell chmod 755 {REMOTE_BIN}")
        adb(serial, "logcat -c")

        blob = {"ts": ts, "device": serial, "mode": "conv2d_taps",
                "tiles": args.tiles, "iters": args.iters, "attempts": [],
                "classes": {c["name"]: c for c in classes}}
        for c in sel:
            got = None
            for t in ladder(c):
                r = run_one(serial, c, t, args.tiles, args.iters, args.timeout)
                blob["attempts"].append(r)
                print(f"{c['name'][:30]:30s} split={t['split']:6s} "
                      f"N={t['N']:2d} {t['H']}x{t['W']} -> rc={r['rc']} "
                      f"fin={r['model_finish_rc']} ms={r['ms']} gops={r['gops']} "
                      f"deleg={r['delegated']} gate={r['gate_ok']}")
                if r["ok"]:
                    got = r
                    break
            if got is None:
                print(f"{c['name']}: FAILED on every decomposition")
            # write incrementally so an interrupted sweep is still usable
            out_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
        return 0

    if args.report:
        files = sorted(DEVICE_DIR.glob("conv_sweep_*.json"))
        if not files:
            print("no sweep artifacts")
            return 2
        blob = json.loads(files[-1].read_text(encoding="utf-8"))
        by = {}
        for a in blob["attempts"]:
            if a.get("ok") and a["class"] not in by:
                by[a["class"]] = a
        total_ms = 0.0
        total_macs = 0
        print(f"{'class':30s} {'ms_measured':>11s} {'subops':>6s} {'layer_ms':>9s} "
              f"{'ops':>3s} {'class_ms':>9s} {'GOPS':>6s}")
        for name, c in blob["classes"].items():
            a = by.get(name)
            if not a:
                print(f"{name[:30]:30s} {'TBD':>11s}")
                continue
            layer_ms = a["ms"] * c["ktaps"] * a["sub_ops"]
            class_ms = layer_ms * c["ops"]
            total_ms += class_ms
            total_macs += c["total_macs"]
            print(f"{name[:30]:30s} {a['ms']:11.3f} {a['sub_ops']:6d} {layer_ms:9.3f} "
                  f"{c['ops']:3d} {class_ms:9.1f} "
                  f"{2*c['total_macs']/(class_ms/1000)/1e9:6.1f}")
        print(f"\nT_V (TPU, CONV_2D taps) = {total_ms/1000:.3f} s/chunk   "
              f"{2*total_macs/(total_ms/1000)/1e9:.1f} GOPS   (MACs {total_macs/1e9:.1f} G)")
        full = json.loads(OPS_JSON.read_text(encoding="utf-8"))["conv_macs"]
        print(f"coverage: {total_macs/1e9:.1f} GMAC of {full/1e9:.1f} GMAC conv work "
              f"({100*total_macs/full:.1f} %) -> T_V is a "
              f"{'COMPLETE' if total_macs >= 0.99*full else 'PARTIAL'} measured sum")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
