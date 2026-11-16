#!/usr/bin/env python3
"""P3.7c — what the *bit-exact* NPU recipe costs (`T_N` on the exact shapes).

P3.7b measured the only configuration this HAL returns exactly: one graph per
384-column output block, row tiles of <= 200 rows (100.00 % bit-exact at
`M = 1/8/128/160/192/200`, and `N = 384 x M = 384` does not even compile). That
recipe needs many more dispatches than the 9-graph stage the pipeline runs
today, so its cost has to be measured before any cut argument touches it.

`proc_b --recipe 1` does the measuring: one compilation per column block (the
weight window is the only per-block constant operand), then each graph is
executed once per row tile, `--recipe-reps` times. This driver gates the run
thermally, runs it, and writes the capture.

    python tools/pipe/tn_recipe.py                    # 9 GEMMs, 5 reps, M=192
    python tools/pipe/tn_recipe.py --reps 8           # tighter stats
    python tools/pipe/tn_recipe.py --gemms 0          # one GEMM, quick sanity
    python tools/pipe/tn_recipe.py --verify-only      # re-parse the last capture

Every number in the doc comes from this run's device output. Nothing modelled.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "probe"))
from probe_host import ADB, adb, pick_serial, read_zones, run  # noqa: E402

PIPE_DIR = REPO / "tools" / "android" / "pipe"
PIPE_BIN = PIPE_DIR / "build" / "proc_b"
ART = REPO / "artifacts" / "blocks"
DEVICE_DIR = REPO / "docs" / "device"

REMOTE_BIN = "/data/local/tmp/hv_proc_b"
REMOTE_BLK = "/data/local/tmp/hv_blk.bin"
REMOTE_DUMP = "/data/local/tmp/hv_recipe"
GATE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")
BLOCK_BYTES = 46_399_488
KD = 1536

# Geometry of a Preset-A block, same table as proc_b/proc_a.
GEMMS = [("attn1.to_q", 1536), ("attn1.to_k", 1536), ("attn1.to_v", 1536),
         ("attn1.to_out", 1536), ("attn2.to_q", 1536), ("attn2.to_k", 1536),
         ("attn2.to_v", 1536), ("attn2.to_out", 1536), ("ffn.up", 8960)]
GEMM_OFF = {"attn1.to_q": 0, "attn1.to_k": 2_359_296, "attn1.to_v": 4_718_592,
            "attn1.to_out": 7_077_888, "attn2.to_q": 9_437_184, "attn2.to_k": 11_796_480,
            "attn2.to_v": 14_155_776, "attn2.to_out": 16_515_072, "ffn.up": 18_874_368,
            "ffn.down": 32_636_928}

# Measured references this run is compared against (cited, never re-derived).
REF_9GRAPH_MS = 377.8          # P3.3, proc_b, M=1530, 9 held graphs (RESULTS_proc_b_p33.md)
REF_9GRAPH_CORUN_MS = 390.4    # co-run, RESULTS_corun_tg.md
REF_P33_SPREAD = (373.4, 381.1)


def kv(line: str) -> dict:
    d: dict = {}
    for part in line.split("|")[1:]:
        k, sep, v = part.partition("=")
        if not sep:
            continue
        if k in ("name", "gemm", "act", "note"):
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


def verify_dumps(dump: Path, blk: Path, row: int, tokens: int, nrow: int,
                 gemm_off: dict[str, int], gemm_k: int = KD) -> dict:
    """Check the recipe's outputs against the weight bytes, element-wise.

    With the `basis` activation (`A[r][r%K] = 127`) and `SA = SW = 1/127`, NNAPI's
    requant gives `out[r][n] = clamp(round(127*W[col0+n][r%K]/127)) = W[col0+n][r%K]`,
    i.e. one stored INT8 weight byte. So exactness needs no BLAS at all: compare
    the dumped tile with the artifact slice `block[off + (col0+n)*K + (r%K)]`.
    """
    import numpy as np

    raw = np.fromfile(blk, dtype=np.int8)
    files = sorted(dump.rglob("g*_c*_t*_out.bin"))
    tiles, exact, worst, bad = 0, 0, 0, []
    per_gemm: dict[str, dict] = {}
    for f in files:
        stem = f.stem.split("_")
        g, c0, t = int(stem[0][1:]), int(stem[1][1:]), int(stem[2][1:])
        nm = GEMMS[g][0]
        nout = GEMMS[g][1]
        n = min(384, nout - c0)
        o = np.fromfile(f, dtype=np.int8)
        if o.size != row * n:
            bad.append({"file": f.name, "error": f"{o.size} != {row*n}"})
            continue
        o = o.reshape(row, n).astype(np.int16)
        # weight rows col0..col0+n-1, columns (r % K) for the tile's global rows
        w = raw[gemm_off[nm]:gemm_off[nm] + nout * gemm_k].reshape(nout, gemm_k)
        expect = np.empty((row, n), dtype=np.int16)
        for r in range(row):
            gr = (t * row + r) % gemm_k
            expect[r, :] = w[c0:c0 + n, gr]
        d = o - expect
        frac = float((d == 0).mean())
        mx = int(np.abs(d).max())
        tiles += 1
        worst = max(worst, mx)
        if mx == 0:
            exact += 1
        else:
            bad.append({"file": f.name, "bit_exact_frac": frac, "max_abs_lsb": mx})
        pg = per_gemm.setdefault(nm, {"tiles": 0, "exact": 0, "max_abs_lsb": 0})
        pg["tiles"] += 1
        pg["exact"] += 1 if mx == 0 else 0
        pg["max_abs_lsb"] = max(pg["max_abs_lsb"], mx)
    return {"files": len(files), "tiles": tiles, "exact_tiles": exact,
            "worst_abs_lsb": worst, "per_gemm": per_gemm,
            "mismatches": bad[:20], "verdict": bool(tiles and exact == tiles)}


def zone_line(z: dict) -> str:
    return " ".join(f"{k}={z[k]:.1f}" for k in GATE_KEYS if k in z)


def gate_wait(serial: str, slack: float, max_wait: float, poll: float, confirm: int) -> tuple[bool, dict]:
    """Block until max(TPU, skin, VIRTUAL-SKIN) <= battery + 3 + slack.

    The TPU zone reports in whole degrees, so a *confirm* sample can land on the
    next step up while the device has not moved. Aborting the run on that single
    sample wasted a floor-grade device (observed 2026-11-16: peak 27.0 PASS, then
    28.4 on the confirm at an unchanged battery). A failed confirm now returns to
    the wait loop instead of failing the gate, still bounded by `max_wait`.
    """
    t0 = time.time()
    while True:
        while True:                      # 1) wait for a passing sample
            z = read_zones(serial)
            peak = max((v for k, v in z.items() if k in GATE_KEYS), default=None)
            batt = z.get("battery")
            if peak is None or batt is None:
                print("  [gate] zones unreadable — refusing a timed run")
                return False, z
            gate = batt + 3.0 + slack
            el = int(time.time() - t0)
            print(f"  [gate {el:4d}s] peak={peak:.1f} gate={gate:.1f}  ({zone_line(z)})")
            if peak <= gate:
                break
            if time.time() - t0 > max_wait:
                print(f"  [gate] gave up after {el}s: peak {peak:.1f} > gate {gate:.1f}")
                return False, z
            time.sleep(poll)
        ok = True                       # 2) confirm on fresh samples
        for i in range(max(1, confirm)):
            time.sleep(8)
            z = read_zones(serial)
            peak = max((v for k, v in z.items() if k in GATE_KEYS), default=None)
            gate = (z.get("battery") or 99) + 3.0 + slack
            print(f"  [gate confirm {i+1}/{max(1, confirm)}] peak={peak:.1f} gate={gate:.1f}")
            if peak is None or peak > gate:
                ok = False
                break
        if ok:
            return True, z
        if time.time() - t0 > max_wait:
            print("  [gate] confirm kept stepping over the gate; giving up")
            return False, z
        print("  [gate] confirm stepped over the gate — back to waiting (device did not move)")
        time.sleep(poll)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemms", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--row", type=int, default=192, help="row tile (measured exact ceiling: 200)")
    ap.add_argument("--col", type=int, default=384, help="column block (measured exact width)")
    ap.add_argument("--tokens", type=int, default=1536, help="tokens per Preset-A block")
    ap.add_argument("--block-idx", type=int, default=0)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--gate-slack", type=float, default=1.0,
                    help="C added to battery+3; 1.0 because the TPU zone reports in whole "
                         "degrees and the idle floor sits between two steps (see cool_rerun.py)")
    ap.add_argument("--max-wait", type=float, default=1800)
    ap.add_argument("--poll", type=float, default=45)
    ap.add_argument("--confirm", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--dump", action="store_true",
                    help="dump every graph's int8 output on pass 1 and verify it against the "
                         "weight bytes on the host (basis activation => expected == W byte)")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    temps_before: dict = {}
    temps_after: dict = {}
    gate_ok = False
    stage = "unknown"
    dump_res = None

    if args.verify_only:
        logs = sorted(DEVICE_DIR.glob("tn_recipe_*.raw.log"))
        if not logs:
            print("no tn_recipe raw log to verify", file=sys.stderr)
            return 2
        raw = logs[-1].read_text(encoding="utf-8")
        serial = "verify-only"
        print(f"verifying {logs[-1].name}")
    else:
        if args.build or not PIPE_BIN.exists():
            print("building proc_b ...")
            rc, txt = run(["bash", (PIPE_DIR / "build.sh").as_posix()], timeout=900)
            if rc != 0:
                print(txt[-3000:], file=sys.stderr)
                return 2
        serial = args.serial or pick_serial()
        if not serial:
            print("no device attached", file=sys.stderr)
            return 2
        blk = ART / f"block_{args.block_idx:02d}.bin"
        if not blk.exists():
            print(f"missing {blk}", file=sys.stderr)
            return 2

        print("  gating the run thermally ...")
        gate_ok, temps_before = gate_wait(serial, args.gate_slack, args.max_wait,
                                          args.poll, args.confirm)
        if not gate_ok:
            peak = max((v for k, v in temps_before.items() if k in GATE_KEYS), default=None)
            batt = temps_before.get("battery")
            print(f"  [gate] NO TIMED RUN: peak {peak} vs battery+3 {None if batt is None else batt + 3}")
            return 3

        rc, txt = run([str(ADB), "-s", serial, "push", str(PIPE_BIN), REMOTE_BIN], timeout=600)
        if rc != 0:
            print(f"push failed: {txt}", file=sys.stderr)
            return 2
        adb(serial, f"shell chmod 755 {REMOTE_BIN}")
        rcs_, out = adb(serial, f"shell stat -c %s {REMOTE_BLK}")
        if out.strip() != str(BLOCK_BYTES):
            print(f"{REMOTE_BLK} is {out.strip()!r} on device, not {BLOCK_BYTES} — pushing")
            rc, txt = run([str(ADB), "-s", serial, "push", str(blk), REMOTE_BLK], timeout=900)
            if rc != 0:
                print(f"push block failed: {txt}", file=sys.stderr)
                return 2
        adb(serial, "logcat -c")      # rule: clear the NNAPI buffer before any graph work

        dump_local = DEVICE_DIR / "recipe_dump" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.dump:
            adb(serial, f"shell rm -rf {REMOTE_DUMP}")

        cmd = (f"shell cd /data/local/tmp && {REMOTE_BIN} --block {REMOTE_BLK} --recipe 1 "
               f"--recipe-gemms {args.gemms} --recipe-reps {args.reps} --recipe-row {args.row} "
               f"--recipe-col {args.col} --recipe-tokens {args.tokens} --down skip --decode 0"
               + (f" --recipe-dump {REMOTE_DUMP}" if args.dump else ""))
        print(f"  running: gemms={args.gemms} reps={args.reps} row={args.row} col={args.col} "
              f"tokens={args.tokens}")
        t0 = time.time()
        rc, raw = adb(serial, cmd, timeout=args.timeout)
        wall = time.time() - t0
        temps_after = read_zones(serial)
        stage = "run"
        print(f"  device wall {wall:.1f} s, rc={rc}")

        dump_res = None
        if args.dump:
            dump_local.mkdir(parents=True, exist_ok=True)
            rc2, txt2 = run([str(ADB), "-s", serial, "pull", REMOTE_DUMP, str(dump_local)], timeout=900)
            n = len(list(dump_local.rglob("*.bin")))
            print(f"  pulled {n} dump file(s) (rc={rc2})")
            if n:
                dump_res = verify_dumps(dump_local, blk, args.row, args.tokens,
                                        (args.tokens + args.row - 1) // args.row, GEMM_OFF)
                print(f"  parity: {dump_res['exact_tiles']}/{dump_res['tiles']} tiles bit-exact, "
                      f"worst |Δ| {dump_res['worst_abs_lsb']} LSB")

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        (DEVICE_DIR / f"tn_recipe_{ts}.raw.log").write_text(raw, encoding="utf-8")

    plan = next((kv(l) for l in lines if l.startswith("RECIPE|plan|")), {})
    gemm_lines = [kv(l) for l in lines if l.startswith("RECIPE|gemm|")]
    col_lines = [kv(l) for l in lines if l.startswith("RECIPE|col|")]
    pass_lines = [kv(l) for l in lines if l.startswith("RECIPE|pass|")]
    summ = [kv(l) for l in lines if l.startswith("SUMMARY|recipe|")]
    fails = [l for l in lines if l.startswith("RECIPE|FAIL") or l.startswith("FATAL")
             or l.startswith("WARN")]

    reps = int(plan.get("reps") or args.reps)
    row = int(plan.get("row") or args.row)
    col = int(plan.get("col") or args.col)
    tokens = int(plan.get("tokens") or args.tokens)
    nrow = (tokens + row - 1) // row

    # per-pass block totals (index 1..reps) from the SUMMARY lines
    block_ms = {int(s["pass"]): float(s["block_ms"]) for s in summ if "pass" in s}
    compile_ms = next((float(s["compile_ms"]) for s in summ if "compile_ms" in s), None)
    graph_total = next((int(s["graphs"]) for s in summ if "graphs" in s), None)

    # per-GEMM per-pass totals
    per_gemm: dict[str, dict] = {}
    for p in pass_lines:
        nm = str(p.get("name"))
        per_gemm.setdefault(nm, {"N_out": None, "passes": {}})
        per_gemm[nm]["passes"][int(p["pass"])] = float(p["ms_tot"])
    for gl in gemm_lines:
        nm = str(gl.get("name"))
        if nm in per_gemm:
            per_gemm[nm]["N_out"] = int(gl["N_out"])
            per_gemm[nm]["col_blocks"] = int(gl["col_blocks"])
            per_gemm[nm]["execs_per_pass"] = int(gl["execs_per_pass"])
            per_gemm[nm]["compile_ms"] = float(gl.get("compile_ms") or 0.0)
            per_gemm[nm]["compile_fails"] = int(gl.get("compile_fails") or 0)
    for nm, d in per_gemm.items():
        vals = [v for k, v in sorted(d["passes"].items()) if k > 0]
        d["ms_block"] = st.median(vals) if vals else None
        d["ms_series"] = vals
        if d["ms_block"] and d.get("N_out"):
            macs = nrow * row * d["N_out"] * KD          # per block, tokens x N_out x K
            d["gops"] = 2.0 * macs / (d["ms_block"] * 1e6)
        else:
            d["gops"] = None

    # measured exec totals and the derived block rate
    execs = sum((d.get("execs_per_pass") or 0) for d in per_gemm.values())
    compiles = sum((d.get("col_blocks") or 0) for d in per_gemm.values())
    macs_block = nrow * row * sum((d.get("N_out") or 0) for d in per_gemm.values()) * KD
    bm_stats = stats([block_ms[k] for k in sorted(block_ms)])
    # steady state = median of passes 2..reps (pass 1 pays the cold weight fetch)
    warm = [block_ms[k] for k in sorted(block_ms) if k >= 2]
    warm_stats = stats(warm)

    peak_b = max((v for k, v in temps_before.items() if k in GATE_KEYS), default=None) if temps_before else None
    batt_b = temps_before.get("battery") if temps_before else None

    res = {
        "timestamp": ts, "serial": serial, "stage": "P3.7c exact-recipe T_N",
        "block_file": f"block_{args.block_idx:02d}.bin", "block_bytes": BLOCK_BYTES,
        "plan": {"row": row, "col": col, "tokens": tokens, "row_tiles": nrow, "reps": reps,
                 "gemms": args.gemms},
        "compiles": compiles, "execs_per_pass": execs,
        "compile_ms": compile_ms,
        "block_ms_per_pass": block_ms,
        "block_ms_stats": bm_stats, "block_ms_warm_stats": warm_stats,
        "macs_per_block": macs_block,
        "gops_warm": (2.0 * macs_block / (warm_stats["median"] * 1e6)) if warm_stats else None,
        "per_gemm": per_gemm,
        "reference_9graph_ms": REF_9GRAPH_MS,
        "reference_9graph_corun_ms": REF_9GRAPH_CORUN_MS,
        "reference_spread_ms": list(REF_P33_SPREAD),
        "ratio_vs_9graph": (warm_stats["median"] / REF_9GRAPH_MS) if warm_stats else None,
        "temperatures": {"before": temps_before, "after": temps_after,
                         "gate_ok": gate_ok, "peak_before_c": peak_b,
                         "battery_before_c": batt_b,
                         "gate": "max(TPU, skin1, skin2, VIRTUAL-SKIN) <= battery + 3 C"},
        "parity": dump_res,
        "fails": fails[:20],
    }
    (DEVICE_DIR / f"tn_recipe_{ts}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ markdown
    def fu(x, f="{:.1f}"):
        return "TBD" if x is None else f.format(x)

    L = [f"# Exact-recipe `T_N` — the cost of the only bit-exact NPU stage — {ts}", "",
         f"serial `{serial}` · `block_{args.block_idx:02d}.bin` ({BLOCK_BYTES:,} B) · "
         f"row tile **M={row}** · column block **N={col}** · **{tokens} tokens/block** "
         f"({nrow} row tiles) · {reps} pass(es)", "",
         "## Verdict", ""]
    if not block_ms:
        L.append("- **NO MEASUREMENT** — no `SUMMARY|recipe` line came back; see the raw log")
    else:
        L += [f"- **exact recipe costs {fu(warm_stats and warm_stats['median'])} ms/block** "
              f"({fu(warm_stats and warm_stats['median'] / 1000, '{:.4f}')} s), "
              f"**{fu(res['ratio_vs_9graph'], '{:.2f}')}× the {REF_9GRAPH_MS} ms 9-graph stage** "
              f"the pipeline runs today (`RESULTS_proc_b_p33.md`)",
              f"- {compiles} compilations / {execs} executes per block, "
              f"compile **{fu(compile_ms, '{:.0f}')} ms** (one-time per process, but per *block* "
              f"if the graphs are bound to that block's weight windows)",
              f"- **{fu(res['gops_warm'], '{:.0f}')} GOPS** over {macs_block / 1e9:.3f} GMAC",
              f"- fails: **{len(fails)}**" + ("" if not fails else " — see the raw log"),
              ""]
    L += ["## Numbers", "", "| quantity | value |", "|---|---|"]
    if block_ms:
        L += [f"| block, pass 1 (cold weight fetch) | {fu(block_ms.get(1))} ms |",
              f"| block, passes 2…{reps} (steady) | **{fu(warm_stats and warm_stats['median'])} ms** "
              f"median (n={warm_stats and warm_stats['n']}, spread "
              f"{fu(warm_stats and warm_stats['min'])}–{fu(warm_stats and warm_stats['max'])}) |",
              f"| block, all passes | {fu(bm_stats and bm_stats['median'])} ms median, "
              f"{fu(bm_stats and bm_stats['mean'])} mean |",
              f"| GOPS (steady) | {fu(res['gops_warm'], '{:.0f}')} |",
              f"| vs the 9-graph stage ({REF_9GRAPH_MS} ms) | "
              f"**{fu(res['ratio_vs_9graph'], '{:.2f}')}×** |",
              f"| vs the co-run 9-GEMM reference ({REF_9GRAPH_CORUN_MS} ms) | "
              f"{fu(warm_stats and warm_stats['median'] / REF_9GRAPH_CORUN_MS, '{:.2f}')}× |",
              f"| compilations per block | {compiles} (one per column block) |",
              f"| executes per block | {execs} ({execs // max(1, nrow)} graphs × {nrow} row tiles) |",
              f"| compile wall, all {compiles} graphs | {fu(compile_ms, '{:.0f}')} ms |",
              ""]
    L += ["### Per GEMM (median over passes 2…%d)" % reps, "",
          "| GEMM | N_out | col blocks | execs/pass | ms/block | GOPS | compile ms |", "|---|---|---|---|---|---|---|"]
    for nm, d in per_gemm.items():
        L.append(f"| `{nm}` | {d.get('N_out')} | {d.get('col_blocks')} | {d.get('execs_per_pass')} | "
                 f"{fu(d['ms_block'], '{:.2f}')} | {fu(d['gops'], '{:.0f}')} | "
                 f"{fu(d.get('compile_ms'), '{:.0f}')} |")
    if dump_res:
        pg = dump_res["per_gemm"]
        L += ["", "## Parity of the recipe (host-checked against the weight bytes)", "",
              f"`basis` activations make the exact output one stored INT8 weight byte, so the "
              f"check needs no BLAS: **{dump_res['exact_tiles']}/{dump_res['tiles']} dumped tiles "
              f"bit-exact (Δ = 0 LSB), worst |Δ| = {dump_res['worst_abs_lsb']} LSB** ⇒ "
              f"**{'PASS' if dump_res['verdict'] else 'FAIL'}**", "",
              "| GEMM | tiles | exact | worst \\|Δ\\| (LSB) |", "|---|---|---|---|"]
        for nm, d in pg.items():
            L.append(f"| `{nm}` | {d['tiles']} | {d['exact']} | {d['max_abs_lsb']} |")
        if dump_res["mismatches"]:
            L += ["", "First mismatching tiles:", ""] + \
                 [f"- `{m}`" for m in dump_res["mismatches"][:8]]
    L += ["", "## Thermal", "",
          "- before: " + (zone_line(temps_before) if temps_before else "not sampled"),
          "- after:  " + (zone_line(temps_after) if temps_after else "not sampled"),
          f"- gate (`max zone <= battery+3`, slack {args.gate_slack:.1f} °C for the whole-degree "
          f"TPU sensor): **{'PASS at start' if gate_ok else 'NOT GATED'}**", "",
          "## What this proves / does not", "",
          "- proves: the measured cost of the only graph geometry this HAL returns bit-exactly "
          "(P3.7b's map), on the real INT8 artifact, over a full Preset-A block, with repeats"
          + (" — and that the whole block's 448 tiles come back bit-exact under repeated "
             "executes, not just the probed shapes" if dump_res and dump_res["verdict"] else ""),
          "- does not prove: block-to-block variation (one block region measured), `ffn.down` "
          "(K=8960 has no graph), the co-run rate (this is a solo stage), or anything about a "
          "per-column-scale NPU graph (the HAL still rejects those)",
          "- the compilation count is per *column block*, not per row tile: a graph's weight "
          "operand is a window into the artifact, and the same M=%d graph is re-executed over "
          "each row tile's activation slice" % row,
          ]
    if fails:
        L += ["", "## Warnings / failures", ""] + [f"- `{f}`" for f in fails[:10]]
    md = DEVICE_DIR / f"tn_recipe_{ts}.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\nwrote {md.name} and tn_recipe_{ts}.json")
    if block_ms:
        print(f"  block: pass1={block_ms.get(1)} ms, steady median="
              f"{(warm_stats or {}).get('median')} ms  -> {res['ratio_vs_9graph']:.2f}x the "
              f"{REF_9GRAPH_MS} ms stage")
    for nm, d in per_gemm.items():
        print(f"  {nm:<12} {d.get('col_blocks'):>3} col blocks  "
              f"{(d.get('execs_per_pass') or 0):>4} execs  {fu(d['ms_block'], '{:7.2f}')} ms  "
              f"{fu(d['gops'], '{:5.0f}')} GOPS")
    return 0 if block_ms and not any(f.startswith(("FATAL", "RECIPE|FAIL")) for f in fails) else 1


if __name__ == "__main__":
    sys.exit(main())
