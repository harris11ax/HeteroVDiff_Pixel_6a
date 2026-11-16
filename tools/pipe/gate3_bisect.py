#!/usr/bin/env python3
"""P3.7b — Gate 3 bisect: is the broken FULLY_CONNECTED path the multi-output-
block one or the ``M = 384`` row one?

Gate 3 (`RESULTS_gate3_parity.md`) measured a `[384, N]` FC tile returning the
exact integer product for **25.8 %** of its elements, and the correct region was
structural: the **last 384-column block** for every row, plus **row 0** for every
column. Two explanations fit and they have different fixes, so this tool builds
the discriminating graphs instead of arguing:

| probe | if EXACT | if NOT exact |
|---|---|---|
| `m384_n384_c0` (one output block) | break is the **multi-block** path | break is the **row** path |
| `m1_nfull` (one row, all blocks) | break is the multi-block path | break is the row path |
| `m384_n768_*` (two blocks) | per-block column vector says which block survives | — |
| `m384_nfull` | control — must reproduce the 25.8 % | harness broken |

Weight windows are `off + col0*K` into the one file-mapped INT8 artifact (no
copies, still route B). Activations are the `basis` pattern `A[r][r%K] = 127`,
so the exact output element is one stored weight byte and nothing saturates.
The host recomputes `acc = Σ a·w` exactly (fp32 BLAS with K-split partials
bounded < 2^24, accumulated in int64) and applies NNAPI's requant.

    python tools/pipe/gate3_bisect.py                 # push, probe, pull, verify
    python tools/pipe/gate3_bisect.py --gemm 8        # same sweep on ffn.up (N=8960)
    python tools/pipe/gate3_bisect.py --verify-only   # re-verify the last pull
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
sys.path.insert(0, str(REPO / "tools" / "pipe"))
from probe_host import ADB, adb, pick_serial, read_zones, run  # noqa: E402
from gate3_parity import MODES, SA, SW, block_gemms, exact_acc, fnv1a, requant  # noqa: E402

PIPE_DIR = REPO / "tools" / "android" / "pipe"
PIPE_BIN = PIPE_DIR / "build" / "proc_b"
ART = REPO / "artifacts" / "blocks"
DEVICE_DIR = REPO / "docs" / "device"
DUMP_LOCAL = REPO / "docs" / "device" / "bisect_dump"

REMOTE_BIN = "/data/local/tmp/hv_proc_b"
REMOTE_BLK = "/data/local/tmp/hv_blk.bin"
REMOTE_PROBE = "/data/local/tmp/hv_probe"
ZONE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")
BLK = 384                       # output-column block the HAL writes correctly


# ------------------------------------------------------------------ device run
def kv_line(line: str) -> dict:
    d = {}
    for part in line.split("|")[1:]:
        k, sep, v = part.partition("=")
        if sep:
            d[k] = v
    return d


def run_on_device(serial: str, gemm: int, plan: str = "") -> tuple[int, str]:
    rc, txt = run([str(ADB), "-s", serial, "push", str(PIPE_BIN), REMOTE_BIN], timeout=600)
    if rc != 0:
        raise SystemExit(f"push failed: {txt}")
    adb(serial, f"shell chmod 755 {REMOTE_BIN}")
    # the artifact is 46,399,488 B; proc_b refuses anything else
    rc, out = adb(serial, f"shell stat -c %s {REMOTE_BLK}")
    if str(out).strip() != "46399488":
        raise SystemExit(f"{REMOTE_BLK} on device is {out.strip()!r}, not 46399488 B — "
                         f"push artifacts/blocks/block_00.bin there first")
    adb(serial, f"shell rm -rf {REMOTE_PROBE}")
    adb(serial, "logcat -c")
    pl = f" --probe-plan '{plan}'" if plan else ""
    cmd = (f"shell cd /data/local/tmp && {REMOTE_BIN} --block {REMOTE_BLK} "
           f"--probe-dir {REMOTE_PROBE} --probe-g {gemm}{pl} --down skip --decode 0")
    return adb(serial, cmd, timeout=420)


def pull_dump(serial: str, local: Path) -> list[Path]:
    if local.exists():
        for p in local.glob("*.bin"):
            p.unlink()
    local.mkdir(parents=True, exist_ok=True)
    rc, out = adb(serial, f"shell ls {REMOTE_PROBE}")
    names = [l.strip() for l in out.splitlines() if l.strip().endswith(".bin")]
    pulled = []
    for nm in names:
        dst = local / nm
        rc, txt = run([str(ADB), "-s", serial, "pull", f"{REMOTE_PROBE}/{nm}", str(dst)], timeout=600)
        if rc != 0 or not dst.exists():
            raise SystemExit(f"pull {nm} failed: {txt}")
        pulled.append(dst)
    return pulled


# -------------------------------------------------------------------- verify
def basis_act(rows: int, k: int) -> np.ndarray:
    a = np.zeros((rows, k), dtype=np.int8)
    for r in range(rows):
        a[r, r % k] = 127
    return a


def verify_probe(probe: dict, raw: np.ndarray, dump: Path) -> dict:
    tag, N, K, M = probe["tag"], int(probe["N"]), int(probe["K"]), int(probe["M"])
    col0, woff = int(probe["col0"]), int(probe["woff"])
    fi = next(dump.glob(f"*_{tag}_in.bin"), None)
    fo = next(dump.glob(f"*_{tag}_out.bin"), None)
    if fi is None or fo is None:
        return {"tag": tag, "N": N, "K": K, "M": M, "col0": col0, "error": "dump missing"}
    a = np.fromfile(fi, dtype=np.int8)
    o = np.fromfile(fo, dtype=np.int8)
    if a.size != M * K:
        return {"tag": tag, "N": N, "K": K, "M": M, "col0": col0,
                "error": f"in {a.size} != {M*K}"}
    if not np.array_equal(a.reshape(M, K), basis_act(M, K)):
        return {"tag": tag, "N": N, "K": K, "M": M, "col0": col0,
                "error": "dumped activation != basis reconstruction (device built something else)"}
    if o.size != M * N:
        return {"tag": tag, "N": N, "K": K, "M": M, "col0": col0,
                "error": f"out {o.size} != {M*N}"}
    a = a.reshape(M, K)
    o = o.reshape(M, N).astype(np.int16)

    wb = raw[woff:woff + N * K]
    if wb.size != N * K:
        return {"tag": tag, "N": N, "K": K, "M": M, "col0": col0,
                "error": f"weight window {wb.size} != {N*K}"}
    w_i8 = wb.view(np.int8).reshape(N, K)

    acc = exact_acc(a, w_i8)
    out = {}
    for md in MODES:
        d = o - requant(acc, md)
        out[md] = {
            "bit_exact_frac": float((d == 0).mean()),
            "max_abs_lsb": int(np.abs(d).max()),
            "n_off_by_2plus": int((np.abs(d) >= 2).sum()),
        }
    best = max(MODES, key=lambda md: (out[md]["bit_exact_frac"], -out[md]["max_abs_lsb"]))

    # where the correct elements are, in 384-column blocks (the structural probe)
    blocks = []
    for b0 in range(0, N, BLK):
        bw = min(BLK, N - b0)
        d = o[:, b0:b0 + bw] - requant(acc[:, b0:b0 + bw], best)
        blocks.append({"col0": col0 + b0, "width": bw,
                       "bit_exact_frac": float((d == 0).mean())})
    # ...and per row, because "row 0 only" was the other half of the signature
    rows = []
    for r in sorted({0, 1, 2, M // 2, M - 1} & set(range(M))):
        d = o[r, :] - requant(acc[r:r + 1, :], best)[0, :]
        rows.append({"row": r, "bit_exact_frac": float((d == 0).mean())})
    inner = o[1:, :] - requant(acc[1:, :], best) if M > 1 else None
    row_min_frac = float((inner == 0).mean()) if inner is not None else None

    return {
        "tag": tag, "N": N, "K": K, "M": M, "col0": col0, "woff": woff,
        "bit_exact_frac": out[best]["bit_exact_frac"],
        "max_abs_lsb": out[best]["max_abs_lsb"],
        "n_off_by_2plus": out[best]["n_off_by_2plus"],
        "best_mode": best, "modes": out,
        "blocks": blocks, "rows": rows,
        "exact_excluding_row0": row_min_frac,
        "fnv_in": f"{fnv1a(a.tobytes()):016x}",
        "fnv_out": f"{fnv1a(o.astype(np.int8).tobytes()):016x}",
    }


def conclude(rows: list[dict], fails: list[str]) -> tuple[str, str]:
    """Turn the probe table into the bisect answer. Nothing is inferred that a
    probe did not measure; a graph that would not compile is reported as such."""
    ok = {r["tag"]: r for r in rows if "error" not in r}
    ftags = {f.split("tag=")[1].split("|")[0] for f in fails if "tag=" in f}
    ncol = [(t, ok[t]["bit_exact_frac"]) for t in ("m1_n384_c0", "m8_n384_c0",
                                                   "m128_n384_c0", "m384_n384_c0")
            if t in ok]
    nfull = [(t, ok[t]["bit_exact_frac"]) for t in ("m1_nfull", "m384_nfull") if t in ok]
    if not ncol and not nfull:
        return "INDETERMINATE", "no probe graph both compiled and ran"
    m384_blk_missing = "m384_n384_c0" in ftags
    m384_full = ok.get("m384_nfull", {}).get("bit_exact_frac")
    m1_full = ok.get("m1_nfull", {}).get("bit_exact_frac")
    m128_blk = ok.get("m128_n384_c0", {}).get("bit_exact_frac")
    m1_blk = ok.get("m1_n384_c0", {}).get("bit_exact_frac")
    m384_blk = ok.get("m384_n384_c0", {}).get("bit_exact_frac")
    detail = ("M=1 N=384 " + (f"{m1_blk*100:.2f}%" if m1_blk is not None else "no graph")
              + " · M=128 N=384 " + (f"{m128_blk*100:.2f}%" if m128_blk is not None else "no graph")
              + " · M=384 N=384 " + ("compile OP_FAILED" if m384_blk_missing
                                     else (f"{m384_blk*100:.2f}%" if m384_blk is not None
                                           else "not probed"))
              + " · M=1 N=full " + (f"{m1_full*100:.2f}%" if m1_full is not None else "no graph")
              + " · M=384 N=full " + (f"{m384_full*100:.2f}%" if m384_full is not None else "no graph"))
    if m1_full is not None and m128_blk is not None and m128_blk >= 0.9999 and m1_full >= 0.9999:
        if m384_blk_missing:
            return ("M384_ROW_TILE",
                    "a 384-column graph is bit-exact at M = 1 and M = 128 and a single row over "
                    "all 1536 columns is bit-exact, while the SAME 384-column graph at M = 384 "
                    "will not even compile — so the defect is the M = 384 row tile, and the "
                    "one-graph-per-column-block fix is unavailable at this M. " + detail)
        if ok.get("m384_n384_c0", {}).get("bit_exact_frac", 0) >= 0.9999:
            return ("MULTI_BLOCK",
                    "a single 384-column block at M = 384 is bit-exact, so the break is the "
                    "multi-output-block path (N > 384). " + detail)
    if m1_full is not None and m1_full < 0.9999 and m384_full is not None and m384_full < 0.9999:
        return ("BOTH",
                f"neither the row path nor the block path is exact (M=1 over all columns is "
                f"{m1_full*100:.2f}%). " + detail)
    return "PARTIAL", "the probes disagree in a way one run does not resolve. " + detail


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemm", type=int, default=0, help="GEMM index 0..8 (0 = attn1.to_q, 8 = ffn.up)")
    ap.add_argument("--block-idx", type=int, default=0)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--plan", default="",
                    help="explicit sweep 'M,N,col0;M,N,col0;...' (default: the discriminating set)")
    ap.add_argument("--tag", default="", help="suffix for the dump dir, so sweeps do not collide")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()

    if args.build or not PIPE_BIN.exists():
        print("building proc_b ...")
        rc, txt = run(["bash", (PIPE_DIR / "build.sh").as_posix()], timeout=900)
        if rc != 0:
            print(txt[-3000:], file=sys.stderr)
            return 2

    dump = DUMP_LOCAL / (f"g{args.gemm}" + (f"_{args.tag}" if args.tag else ""))
    temps_before = temps_after = {}
    raw_log = ""

    if not args.verify_only:
        serial = args.serial or pick_serial()
        if not serial:
            print("no device attached", file=sys.stderr)
            return 2
        temps_before = read_zones(serial)
        rc, raw_log = run_on_device(serial, args.gemm, args.plan)
        pull_dump(serial, dump)
        temps_after = read_zones(serial)
    else:
        serial = args.serial or "verify-only"
        logs = sorted(DEVICE_DIR.glob("gate3_bisect_*.raw.log"))
        if not logs:
            print("no gate3_bisect raw log to verify", file=sys.stderr)
            return 2
        raw_log = logs[-1].read_text(encoding="utf-8")
        print(f"verifying {logs[-1].name}")

    probes = [kv_line(l) for l in raw_log.splitlines()
              if l.startswith("PROBE|") and "tag=" in l and "N=" in l]
    fails = [l for l in raw_log.splitlines() if l.startswith("PROBE|FAIL")]
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if raw_log:
        (DEVICE_DIR / f"gate3_bisect_{ts}.raw.log").write_text(raw_log, encoding="utf-8")

    gemms = block_gemms(args.block_idx)
    spec = gemms[args.gemm]
    raw = np.fromfile(ART / f"block_{args.block_idx:02d}.bin", dtype=np.uint8)
    rows = [verify_probe(p, raw, dump) for p in probes] if probes else []
    if args.plan:
        verdict, why = ("SWEEP", f"explicit sweep of {len(rows)} shapes — read the table, not a "
                                 f"bisect verdict")
    else:
        verdict, why = conclude(rows, fails)

    peak = max((v for k, v in temps_before.items() if k in ZONE_KEYS), default=None)
    batt = temps_before.get("battery")
    res = {
        "timestamp": ts, "serial": serial, "block": args.block_idx,
        "gemm": spec["name"], "gemm_index": args.gemm,
        "N_out": spec["N"], "K": spec["K"],
        "act": "basis (A[r][r%K]=127 => out[r][n] = W[col0+n][r%K])",
        "scales": {"SA": SA, "SW": SW, "note": "per-tensor; per-channel rejected by FULLY_CONNECTED"},
        "probes": rows, "compile_or_exec_failures": fails,
        "temps_before": temps_before, "temps_after": temps_after,
        "thermal_gate": (peak is not None and batt is not None and peak <= batt + 3.0),
        "verdict": verdict, "verdict_note": why,
    }
    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    (DEVICE_DIR / f"gate3_bisect_{ts}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    good = [r for r in rows if "error" not in r]
    L = [f"# Gate 3 bisect — where the FULLY_CONNECTED tile breaks — {ts}", "",
         f"serial `{serial}` · `{spec['name']}` (N_out {spec['N']}, K {spec['K']}) · "
         f"block `block_{args.block_idx:02d}.bin` · basis activations · per-tensor scales "
         f"SA = SW = 1/127", "",
         "## Verdict", "", f"- **{verdict}** — {why}", ""]
    if fails:
        L.append(f"- **{len(fails)} probe graph(s) failed to compile or execute** — see the raw log")
        L.append("")
    L += ["| probe | M | N | col0 | bit-exact | max abs Δ (LSB) | off by ≥2 | exact excl. row 0 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            L.append(f"| `{r['tag']}` | {r['M']} | {r['N']} | {r['col0']} | — | — | — | {r['error']} |")
            continue
        excl = "—" if r["exact_excluding_row0"] is None else f"{r['exact_excluding_row0']*100:.4f}%"
        L.append(f"| `{r['tag']}` | {r['M']} | {r['N']} | {r['col0']} | "
                 f"**{r['bit_exact_frac']*100:.4f}%** | {r['max_abs_lsb']} | "
                 f"{r['n_off_by_2plus']} | {excl} |")
    multi = [r for r in good if len(r["blocks"]) > 1]
    if multi:
        L += ["", "## Correctness per 384-column block (the structural signature)", ""]
        for r in multi:
            vec = " · ".join(f"c{b['col0']}: {b['bit_exact_frac']*100:.2f}%" for b in r["blocks"])
            L.append(f"- `{r['tag']}` (M={r['M']}, N={r['N']}): {vec}")
    L += ["", "## Thermal (parity probe, not a timing table)", "",
          "- before: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_before.items())
                                  if k in ZONE_KEYS),
          "- after:  " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_after.items())
                                  if k in ZONE_KEYS),
          "", "## What this proves / does not", "",
          "- proves: which graph shapes return the exact integer product on this HAL, measured "
          "element-wise against exact integer accumulation over the real INT8 weight bytes",
          "- does **not** prove: why the HAL behaves this way (no readable TPU internals), nor "
          "that a per-block re-graph is fast enough — its cost must be re-measured as `T_N`",
          ]
    md = DEVICE_DIR / f"gate3_bisect_{ts}.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"verdict={verdict} — {why}")
    for r in rows:
        if "error" in r:
            print(f"  {r['tag']:<22} ERROR {r['error']}")
            continue
        print(f"  {r['tag']:<22} M={r['M']:<4} N={r['N']:<5} c0={r['col0']:<5} "
              f"exact={r['bit_exact_frac']*100:8.4f}%  max|d|={r['max_abs_lsb']:<4} "
              f"off2+={r['n_off_by_2plus']}")
    print(f"wrote {md.relative_to(REPO)}")
    return 0 if verdict in ("MULTI_BLOCK",) else 1


if __name__ == "__main__":
    sys.exit(main())
