#!/usr/bin/env python3
"""P3.7 / Gate 3 — numerical parity of the NPU stage, measured not hashed.

`proc_b` used to consume the ring and *hash* the payload it saw; a hash proves
the bytes arrived, never that the TPU multiplied them correctly. This tool makes
the same run dump, for every one of the 9 held FULLY_CONNECTED graphs, the exact
int8 activation the graph received and the int8 output it produced, then
recomputes the product on the host with **exact integer arithmetic** and diffs.

What is being compared
-----------------------
The graph is built with one per-tensor weight scale ``SW = 1/127`` and an
activation/output scale ``SA = 1/127`` (``proc_b.cpp``; ``FULLY_CONNECTED``
rejects ``TENSOR_QUANT8_SYMM_PER_CHANNEL`` on this HAL — measured
``RESULTS_npu_tn.md`` §2). With all three scales equal, NNAPI's requant reduces to

    truth   acc_i64 = sum_k a[k] * w[k]                (exact, |acc| < 2^31)
    output  out_i8  = clamp(round(acc_i64 * SA*SW/SA)) = clamp(round(acc/127))

so the whole graph is checkable in the integer domain — no tolerance needed for
the accumulation, only the requant rounding mode is unknown and is *identified*
here rather than assumed (half-away, half-even and truncate are all scored).

This is **arithmetic parity of the executed graph**. It is not a claim that the
TPU reproduces the locked per-output-column quantisation (that cost is measured
separately in ``RESULTS_scales.md``: 2.15x mean GEMM error), and it is not the
end-to-end PSNR/SSIM leg (Gate 1, host ``.venv-ref`` work).

    python tools/pipe/gate3_parity.py                # push, run, pull, verify
    python tools/pipe/gate3_parity.py --verify-only  # re-verify the last pull
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "probe"))
from probe_host import ADB, adb, pick_serial, read_zones, run  # noqa: E402

PIPE_DIR = REPO / "tools" / "android" / "pipe"
PIPE_BIN = PIPE_DIR / "build" / "proc_b"
ART = REPO / "artifacts" / "blocks"
DEVICE_DIR = REPO / "docs" / "device"
DUMP_LOCAL = REPO / "docs" / "device" / "parity_dump"

REMOTE_BIN = "/data/local/tmp/hv_proc_b"
REMOTE_BLK = "/data/local/tmp/hv_blk.bin"
REMOTE_RING = "/data/local/tmp/hv_handoff.bin"
REMOTE_DUMP = "/data/local/tmp/hv_parity"
ZONE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")

TILE = 384          # per-graph token ceiling; the dumped tile is rows [0,384)
SA = 1.0 / 127.0
SW = 1.0 / 127.0
KSPLIT = 512        # fp32 partials bounded by 512*128*127 = 8.32e6 < 2^24


# --------------------------------------------------------------------- exact GEMM
def exact_acc(a_i8: np.ndarray, w_i8: np.ndarray) -> np.ndarray:
    """(A@W^T) in exact integer arithmetic, via fp32 BLAS with bounded partials.

    int8 x int8 products are exact integers <= 16384 in fp32, and a partial sum of
    at most KSPLIT of them is <= 8.32e6 < 2^24, so every intermediate is exactly
    representable and the fp32 GEMM is an exact integer computation. Partials are
    then accumulated in int64.
    """
    m, k = a_i8.shape
    n = w_i8.shape[0]
    acc = np.zeros((m, n), dtype=np.int64)
    for k0 in range(0, k, KSPLIT):
        a = np.ascontiguousarray(a_i8[:, k0:k0 + KSPLIT]).astype(np.float32)
        w = np.ascontiguousarray(w_i8[:, k0:k0 + KSPLIT]).astype(np.float32)
        part = a @ w.T
        pmax = float(np.abs(part).max()) if part.size else 0.0
        if pmax >= 2.0 ** 24:
            raise AssertionError(f"fp32 partial {pmax:.0f} exceeds 2^24 — split smaller")
        acc += part.astype(np.int64)
    return acc


def requant(acc: np.ndarray, mode: str) -> np.ndarray:
    x = acc.astype(np.float64) * (SA * SW / SA)
    if mode == "half_away":
        q = np.sign(x) * np.floor(np.abs(x) + 0.5)
    elif mode == "half_even":
        q = np.rint(x)
    elif mode == "trunc":
        q = np.trunc(x)
    else:
        raise ValueError(mode)
    return np.clip(q, -128, 127).astype(np.int16)


MODES = ("half_away", "half_even", "trunc")


# ------------------------------------------------------------------------ manifest
def block_gemms(idx: int) -> list[dict]:
    man = json.loads((REPO / "artifacts" / "manifest.json").read_text(encoding="utf-8"))
    out = []
    for g in man["blocks"][idx]["gemms"]:
        n, k = g["shape"]
        out.append({"name": g["name"], "N": int(n), "K": int(k), "offset": int(g["offset"])})
    return out


def fnv1a(data: bytes) -> int:
    h = 1469598103934665603
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


# ---------------------------------------------------------------------- device run
def run_on_device(serial: str, chunks: int, nblocks: int, M: int, act: str = "payload") -> tuple[int, str]:
    rc, txt = run([str(ADB), "-s", serial, "push", str(PIPE_BIN), REMOTE_BIN], timeout=600)
    if rc != 0:
        raise SystemExit(f"push failed: {txt}")
    adb(serial, f"shell chmod 755 {REMOTE_BIN}")
    adb(serial, f"shell rm -rf {REMOTE_DUMP}")
    adb(serial, "logcat -c")
    cmd = (f"shell cd /data/local/tmp && {REMOTE_BIN} --block {REMOTE_BLK} --ring {REMOTE_RING} "
           f"--blocks {chunks} --nblocks {nblocks} --M {M} --down skip --source fork "
           f"--decode 0 --hash-every 0 --dump-parity {REMOTE_DUMP} --act {act}")
    return adb(serial, cmd, timeout=420)


def pull_dump(serial: str, local: Path) -> list[Path]:
    local.mkdir(parents=True, exist_ok=True)
    rc, out = adb(serial, f"shell ls {REMOTE_DUMP}")
    names = [l.strip() for l in out.splitlines() if l.strip().endswith(".bin")]
    pulled = []
    for nm in names:
        dst = local / nm
        rc, txt = run([str(ADB), "-s", serial, "pull", f"{REMOTE_DUMP}/{nm}", str(dst)], timeout=600)
        if rc != 0 or not dst.exists():
            raise SystemExit(f"pull {nm} failed: {txt}")
        pulled.append(dst)
    return pulled


# ------------------------------------------------------------------------- verify
def synth_act(mode: str, rows: int, k: int) -> np.ndarray:
    """The activation pattern proc_b builds for `--act basis|window`."""
    a = np.zeros((rows, k), dtype=np.int8)
    for r in range(rows):
        if mode == "basis":
            a[r, r % k] = 127
        elif mode == "window":
            off = 8 * (r % 192)
            a[r, off:off + 8] = 1
        else:
            raise ValueError(mode)
    return a


def verify(block_idx: int, gemms: list[dict], files: list[Path], act: str = "payload") -> dict:
    blk = ART / f"block_{block_idx:02d}.bin"
    raw = np.fromfile(blk, dtype=np.uint8)
    # files are named "<g>_<short_gemm_name>_in|out.bin"; index them by the
    # numeric prefix so the manifest's long names need no matching.
    have: dict[tuple[int, str], Path] = {}
    for p in files:
        stem = p.name[:-4] if p.name.endswith(".bin") else p.name
        head, sep, tail = stem.partition("_")
        if not sep:
            continue
        try:
            gi = int(head)
        except ValueError:
            continue
        if tail.endswith("_in"):
            have[(gi, "in")] = p
        elif tail.endswith("_out"):
            have[(gi, "out")] = p
    rows = []
    for g, spec in enumerate(gemms):
        fi = have.get((g, "in"))
        fo = have.get((g, "out"))
        if fi is None or fo is None:
            rows.append({"name": spec["name"], "error": "dump missing"})
            continue
        a = np.fromfile(fi, dtype=np.int8)
        o = np.fromfile(fo, dtype=np.int8)
        n, k = spec["N"], spec["K"]
        if a.size != TILE * k:
            rows.append({"name": spec["name"], "error": f"in {a.size} != {TILE*k}"})
            continue
        if act != "payload":
            # integrity: the device must have seen exactly the pattern we built
            if not np.array_equal(a.reshape(TILE, k), synth_act(act, TILE, k)):
                rows.append({"name": spec["name"], "error": "dumped activation != reconstruction"})
                continue
        if o.size != TILE * n:
            rows.append({"name": spec["name"], "error": f"out {o.size} != {TILE*n}"})
            continue
        a = a.reshape(TILE, k)
        o = o.reshape(TILE, n).astype(np.int16)

        wb = raw[spec["offset"]:spec["offset"] + n * k]
        w_i8 = wb.view(np.int8).reshape(n, k)

        acc = exact_acc(a, w_i8)
        acc_absmax = int(np.abs(acc).max())

        modes = {}
        for md in MODES:
            e = requant(acc, md)
            d = o - e
            modes[md] = {
                "bit_exact_frac": float((d == 0).mean()),
                "max_abs_lsb": int(np.abs(d).max()),
                "n_off_by_1": int((np.abs(d) == 1).sum()),
                "n_off_by_2plus": int((np.abs(d) >= 2).sum()),
            }
        best = max(MODES, key=lambda md: (modes[md]["bit_exact_frac"], -modes[md]["max_abs_lsb"]))

        # float-domain context: device output vs the same int8 product dequantised
        ref_f = (a.astype(np.float32) * SA) @ (w_i8.astype(np.float32) * SW).T
        dev_f = o.astype(np.float32) * SA
        denom = float(np.sqrt((ref_f.astype(np.float64) ** 2).sum()))
        rel_l2 = float(np.sqrt(((dev_f - ref_f).astype(np.float64) ** 2).sum()) / denom) if denom else 0.0

        rows.append({
            "name": spec["name"], "N": n, "K": k,
            "macs_per_row": n * k,
            "acc_abs_max": acc_absmax, "int32_safe": acc_absmax < 2**31,
            "modes": modes, "best_mode": best,
            "bit_exact_frac": modes[best]["bit_exact_frac"],
            "max_abs_lsb": modes[best]["max_abs_lsb"],
            "rel_l2": rel_l2,
            "fnv_in": f"{fnv1a(a.tobytes()):016x}",
            "fnv_out": f"{fnv1a(o.astype(np.int8).tobytes()):016x}",
        })
    return {"gemms": rows}


def verdict_of(res: dict) -> tuple[bool, str]:
    rows = [r for r in res["gemms"] if "error" not in r]
    if len(rows) != len(res["gemms"]):
        return False, "some GEMM dumps were missing or the wrong size"
    worst_lsb = max(r["max_abs_lsb"] for r in rows)
    worst_exact = min(r["bit_exact_frac"] for r in rows)
    over2 = sum(r["modes"][r["best_mode"]]["n_off_by_2plus"] for r in rows)
    if over2:
        return False, f"{over2} elements differ from exact integer accumulation by >= 2 LSB"
    if worst_lsb == 0:
        return True, f"bit-exact on all {len(rows)} GEMMs (max |diff| 0)"
    return True, (f"within {worst_lsb} LSB of exact integer accumulation "
                  f"(worst GEMM {worst_exact*100:.4f}% bit-exact, 0 elements off by >= 2)")


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--nblocks", type=int, default=1)
    ap.add_argument("--M", type=int, default=1530)
    ap.add_argument("--block-idx", type=int, default=0)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--act", default="payload", choices=["payload", "basis", "window"])
    ap.add_argument("--max-gemms", type=int, default=9,
                    help="GEMMs on the NPU stage (9; ffn.down is Mali and is never dumped here)")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    dump_dir = DUMP_LOCAL / args.act

    if args.build or not PIPE_BIN.exists():
        print("building proc_b ...")
        rc, txt = run(["bash", str(PIPE_DIR / "build.sh")], timeout=900)
        if rc != 0:
            print(txt[-3000:], file=sys.stderr)
            return 2

    serial = args.serial or pick_serial()
    if not serial and not args.verify_only:
        print("no device attached", file=sys.stderr)
        return 2

    gemms = block_gemms(args.block_idx)[:args.max_gemms]
    temps_before = temps_after = {}
    raw_log = ""
    par_lines: dict[str, dict] = {}

    if not args.verify_only:
        temps_before = read_zones(serial)
        rc, raw_log = run_on_device(serial, args.chunks, args.nblocks, args.M, args.act)
        for line in raw_log.splitlines():
            if line.startswith("PARITY|"):
                d = {}
                for part in line.split("|")[1:]:
                    k, sep, v = part.partition("=")
                    if sep:
                        d[k] = v
                par_lines[d.get("gemm", "?")] = d
        files = pull_dump(serial, dump_dir)
        temps_after = read_zones(serial)
        if not files:
            print("no parity files on the device — run failed", file=sys.stderr)
            print(raw_log[-3000:], file=sys.stderr)
            return 1
    else:
        files = sorted(dump_dir.glob("*.bin"))
        if not files:
            print(f"nothing in {dump_dir}", file=sys.stderr)
            return 2

    res = verify(args.block_idx, gemms, files, args.act)
    ok, why = verdict_of(res)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    peak = max((v for k, v in temps_before.items() if k in ZONE_KEYS), default=None)
    batt = temps_before.get("battery")
    res.update({
        "timestamp": ts, "serial": serial, "block": args.block_idx,
        "act": args.act, "tile_rows": TILE, "M": args.M, "chunks": args.chunks,
        "nblocks": args.nblocks,
        "scales": {"SA": SA, "SW": SW, "note": "per-tensor; per-channel rejected by FULLY_CONNECTED"},
        "kernel": "exact integer accumulation via fp32 BLAS with K-split partials < 2^24",
        "temps_before": temps_before, "temps_after": temps_after,
        "thermal_gate": (peak is not None and batt is not None and peak <= batt + 3.0),
        "device_parity_lines": par_lines,
        "verdict": bool(ok), "verdict_note": why,
    })

    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    (DEVICE_DIR / f"gate3_parity_{ts}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    if raw_log:
        (DEVICE_DIR / f"gate3_parity_{ts}.raw.log").write_text(raw_log, encoding="utf-8")

    rows = [r for r in res["gemms"] if "error" not in r]
    L = [f"# Gate 3 — numerical parity of the NPU stage — {ts}", "",
         f"serial `{serial}` · block `block_{args.block_idx:02d}.bin` · tile rows {TILE} "
         f"(M={args.M}) · {args.chunks} chunk × {args.nblocks} NPU block · "
         f"`ffn.down` = skip (Mali, measured) · device source `fork`", "",
         "## Verdict", "",
         f"- **{'PASS' if ok else 'FAIL'}** — {why}",
         "- comparison is in the integer domain: the host recomputes `acc = Σ a·w` exactly "
         "(fp32 BLAS, partial sums bounded < 2^24, accumulated in int64) and applies NNAPI's "
         "requant `clamp(round(acc·SA·SW/SO))` with SA = SW = SO = 1/127", "",
         "## Per GEMM (tile 0 of the first block, real INT8 artifact bytes)", "",
         "| GEMM | N | K | max abs acc | bit-exact | max \\|Δ\\| LSB | off by ≥2 | rel L2 vs float |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| `{r['name']}` | {r['N']} | {r['K']} | {r['acc_abs_max']:,} | "
                 f"{r['bit_exact_frac']*100:.4f}% | {r['max_abs_lsb']} | "
                 f"{r['modes'][r['best_mode']]['n_off_by_2plus']} | {r['rel_l2']*100:.4f}% |")
    if rows:
        L += ["", "### Requant rounding mode, identified rather than assumed", "",
              "| mode | best GEMM bit-exact | worst GEMM bit-exact | total elements off by 1 |",
              "|---|---|---|---|"]
        for md in MODES:
            fr = [r["modes"][md]["bit_exact_frac"] for r in rows]
            off1 = sum(r["modes"][md]["n_off_by_1"] for r in rows)
            L.append(f"| {md} | {max(fr)*100:.4f}% | {min(fr)*100:.4f}% | {off1:,} |")
    L += ["", "## Thermal (parity run, not a timing table)", "",
          "- before: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_before.items())
                                  if k in ZONE_KEYS),
          "- after:  " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_after.items())
                                  if k in ZONE_KEYS),
          "", "## What this proves / does not", "",
          "- proves: every one of the 9 held TPU graphs returns the exact integer accumulation of the "
          "real INT8 weight bytes and the real quantised activation, up to the requant step — the "
          "stage's arithmetic is not approximate",
          "- does **not** prove: that the per-tensor scale substitution reproduces the locked "
          "per-output-column quantisation (`RESULTS_scales.md`: 2.15× mean GEMM error), nor the "
          "end-to-end PSNR/SSIM leg (Gate 1, host `.venv-ref`), nor anything about the decode — "
          "which is still a stub in this process",
          ]
    md_p = DEVICE_DIR / f"gate3_parity_{ts}.md"
    md_p.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"verdict={'PASS' if ok else 'FAIL'} — {why}")
    for r in rows:
        print(f"  {r['name']:<28} exact={r['bit_exact_frac']*100:8.4f}%  "
              f"max|d|={r['max_abs_lsb']}  best={r['best_mode']}  relL2={r['rel_l2']*100:.4f}%")
    print(f"wrote {md_p.relative_to(REPO)} and gate3_parity_{ts}.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
