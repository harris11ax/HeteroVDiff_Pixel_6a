#!/usr/bin/env python3
"""P3.3 — the NPU stage over the real INT8 artifact (host side).

Pushes `tools/android/pipe/proc_b` and runs it as the consumer half of the
two-slot ring: it waits on `seq`, runs the `(180 - k)` blocks the cut assigns to
the NPU as held FULLY_CONNECTED graphs, then the decode stub, then releases the
slot with `done`.

    python tools/pipe/proc_b.py --chunks 3 --down skip       # default branch
    python tools/pipe/proc_b.py --chunks 3 --down stage      # price the TPU down path
    python tools/pipe/proc_b.py --chunks 3 --source none     # expects a real proc_a

Weight path: one `ANeuralNetworksMemory_createFromFd` view of block_00.bin and a
per-GEMM `setOperandValueFromMemory` window — the same page cache proc_a imports,
so still exactly one stored INT8 copy.

Writes docs/device/proc_b_<ts>.{json,md}. Every number in the doc comes from this
run's device output; anything the run did not measure is written as TBD with a
reason, never modelled.
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
PIPE_BIN = PIPE_DIR / "build" / "proc_b"
ART = REPO / "artifacts" / "blocks"
DEVICE_DIR = REPO / "docs" / "device"

REMOTE_BIN = "/data/local/tmp/hv_proc_b"
REMOTE_BLK = "/data/local/tmp/hv_blk.bin"
REMOTE_SCL = "/data/local/tmp/hv_blk.scales.f32"
REMOTE_RING = "/data/local/tmp/hv_handoff.bin"
ZONE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")
KD, NUP = 1536, 8960
MALI_DOWN_MS = 47.33          # measured, RESULTS_gemm_mali.md §3-4 (cited, not re-derived)


def kv(line: str) -> dict:
    d: dict = {}
    for part in line.split("|")[1:]:
        k, sep, v = part.partition("=")
        if not sep:
            continue
        if k in ("name", "path", "gemm", "branch", "route", "note", "src"):
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


def remote_size(serial: str, path: str) -> int | None:
    rc, out = adb(serial, f"shell stat -c %s {path}")
    try:
        return int(out.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--nblocks", type=int, default=1,
                    help="NPU blocks per chunk (the cut's 180 - k). 1 keeps a unit test bounded.")
    ap.add_argument("--M", type=int, default=1530)
    ap.add_argument("--down", default="skip", choices=["skip", "stage"])
    ap.add_argument("--source", default="fork", choices=["fork", "none"])
    ap.add_argument("--block-idx", type=int, default=0)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--serial", default=None)
    args = ap.parse_args()

    if args.build or not PIPE_BIN.exists():
        print("building proc_b ...")
        rc, txt = run(["bash", str(PIPE_DIR / "build.sh")], timeout=900)
        if rc != 0:
            print(txt[-3000:], file=sys.stderr)
            raise SystemExit("build failed")
        print("build ok")

    blk = ART / f"block_{args.block_idx:02d}.bin"
    scl = ART / f"block_{args.block_idx:02d}.scales.f32"
    for p in (blk, scl, PIPE_BIN):
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

    rc, txt = run([str(ADB), "-s", serial, "push", str(PIPE_BIN), REMOTE_BIN], timeout=600)
    if rc != 0:
        print(f"push failed: {txt}", file=sys.stderr)
        return 2
    adb(serial, f"shell chmod 755 {REMOTE_BIN}")

    # The 46 MB artifact is already on the device from P3.2; only re-push on a
    # size mismatch (a push is ~10 s of USB and pointless when it is identical).
    if remote_size(serial, REMOTE_BLK) != blk.stat().st_size:
        rc, txt = run([str(ADB), "-s", serial, "push", str(blk), REMOTE_BLK], timeout=900)
        if rc != 0:
            print(f"push block failed: {txt}", file=sys.stderr)
            return 2
    if remote_size(serial, REMOTE_SCL) != scl.stat().st_size:
        run([str(ADB), "-s", serial, "push", str(scl), REMOTE_SCL], timeout=600)

    adb(serial, "logcat -c")          # rule: clear the NNAPI buffer before any graph work

    cmd = (f"shell cd /data/local/tmp && {REMOTE_BIN} --block {REMOTE_BLK} --scales {REMOTE_SCL} "
           f"--ring {REMOTE_RING} --blocks {args.chunks} --nblocks {args.nblocks} --M {args.M} "
           f"--down {args.down} --source {args.source} --decode 1")
    print(f"  running: chunks={args.chunks} nblocks={args.nblocks} M={args.M} "
          f"down={args.down} source={args.source}")
    rc, out = adb(serial, cmd, timeout=args.timeout)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    raw_log = DEVICE_DIR / f"proc_b_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.raw.log"
    raw_log.write_text(out, encoding="utf-8")

    blk_lines = [kv(l) for l in lines if l.startswith("BLK|npu|")]
    npu_ts = [kv(l) for l in lines if l.startswith("TS|npu|")]
    prod_ts = [kv(l) for l in lines if l.startswith("TS|prod|")]
    compiles = [kv(l) for l in lines if l.startswith("COMPILE|")]
    downline = next((kv(l) for l in lines if l.startswith("DOWN|")), {})
    hdr = next((kv(l) for l in lines if l.startswith("HV|proc_b|")), {})
    ready = next((kv(l) for l in lines if l.startswith("READY|")), {})
    decode = next((kv(l) for l in lines if l.startswith("DECODE|")), {})
    throughput = next((kv(l) for l in lines if l.startswith("THROUGHPUT|")), {})
    summ = {l.split("|")[1]: kv(l) for l in lines if l.startswith("SUMMARY|")}
    result = next((kv(l) for l in lines if l.startswith("RESULT|")), {})
    fatal = [l for l in lines if l.startswith("FATAL")]
    warns = [l for l in lines if l.startswith("WARN") or l.startswith("EXEC_ERR")]
    temps_after = read_zones(serial)

    per_gemm: dict[str, list[float]] = {}
    for b in blk_lines:
        nm = b.get("gemm")
        if isinstance(nm, str) and isinstance(b.get("ms"), (int, float)):
            per_gemm.setdefault(nm, []).append(float(b["ms"]))
    pg = {n: stats(v) for n, v in per_gemm.items()}

    blocks = [t for t in npu_ts if "block" in t]
    block_ms = [float(t["ms"]) for t in blocks if isinstance(t.get("ms"), (int, float))]
    lat_us = [float(t["lat_ns"]) / 1e3 for t in npu_ts if isinstance(t.get("lat_ns"), (int, float))]
    quant_ms = [float(t["quant_ms"]) for t in npu_ts if isinstance(t.get("quant_ms"), (int, float))]

    nsum = summ.get("npu", {})
    psum = summ.get("prod", {})
    prod_chain = psum.get("hash_chain")
    npu_chain = nsum.get("hash_chain")
    chain_match = (prod_chain is not None and npu_chain is not None and prod_chain == npu_chain) \
        if args.source == "fork" else None

    expect_execs = args.chunks * args.nblocks * 9 * 4          # 4 token tiles at M=1530
    verdict = (bool(result) and result.get("proto_err", 1) == 0
               and nsum.get("gemm_execs") == expect_execs
               and chain_match in (True, None)
               and nsum.get("nan_words", 0) == 0
               and rc == 0
               and not fatal)

    g9 = float(nsum.get("gemms9_ms_per_block") or 0.0)
    macs9 = 4 * 384 * (8 * KD * KD + KD * NUP)                  # 9-GEMM subset, per block
    down_stage_wall = float(nsum.get("down_ms_total") or 0.0)
    per_block_down = down_stage_wall / max(1, args.chunks * args.nblocks) if down_stage_wall else None

    res = {
        "serial": serial,
        "timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "stage": "P3.3 proc_b (NPU stage over the real INT8 artifact)",
        "M": args.M, "chunks": args.chunks, "nblocks": args.nblocks,
        "down": args.down, "source": args.source,
        "block_file": blk.name, "block_bytes": blk.stat().st_size,
        "header": hdr, "ready": ready, "compiles": compiles, "down_line": downline,
        "decode_stub": decode,
        "per_gemm_ms": {n: {"N": next((b.get("N") for b in blk_lines if b.get("gemm") == n), None),
                            "K": next((b.get("K") for b in blk_lines if b.get("gemm") == n), None),
                            **s} for n, s in pg.items()},
        "gemms9_ms_per_block": g9,
        "gemms9_gops": (2.0 * macs9 / (g9 * 1e6)) if g9 else None,
        "block_ms": stats(block_ms),
        "block_series_ms": block_ms,
        "quant_ms": stats(quant_ms),
        "publish_to_consume_us": stats(lat_us),
        "down_ms_per_block_stage": per_block_down,
        "down_mali_ref_ms": MALI_DOWN_MS,
        "gemm_execs": nsum.get("gemm_execs"), "gemm_execs_expected": expect_execs,
        "hash_chain_prod": prod_chain, "hash_chain_npu": npu_chain, "chain_match": chain_match,
        "summary": summ, "throughput": throughput, "result_line": result,
        "fatal": fatal[:10], "warnings": warns[:20], "rc": rc,
        "temps": {"before": temps_before, "after": temps_after,
                  "peak_before_c": peak, "battery_before_c": batt, "gate_ok": gate_ok,
                  "gate": "max(TPU, skin1, skin2, VIRTUAL-SKIN) <= battery + 3 C"},
        "verdict": bool(verdict),
    }

    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    js = DEVICE_DIR / f"proc_b_{res['timestamp']}.json"
    js.write_text(json.dumps(res, indent=2), encoding="utf-8")

    def fu(x, fmt="{:.1f}"):
        return "TBD" if x is None else fmt.format(x)

    bm = res["block_ms"] or {}
    down_tpu_cell = fu(per_block_down, "{:.2f}") if per_block_down else "not run this tick"
    down_ratio_cell = f"{(per_block_down / MALI_DOWN_MS):.2f}×" if per_block_down else "—"
    down_state_cell = ("measured above" if per_block_down
                       else "measured separately: 258.0 ms (`RESULTS_npu_tn.md` §4)")
    down_probe_cell = (fu(per_block_down, "{:.1f}") + " ms per block" if per_block_down
                       else "not executed (branch=skip)")
    L = [f"# P3.3 — NPU stage over the real INT8 artifact — {res['timestamp']}", "",
         f"serial `{serial}` · block region `{blk.name}` ({res['block_bytes']:,} B) · "
         f"M={args.M} (4 token tiles of 384) · {args.chunks} chunk(s) × {args.nblocks} NPU block(s) · "
         f"`ffn.down` = **{args.down}** · source **{args.source}**", "",
         "## Verdict", "",
         f"- **{'PASS' if verdict else 'FAIL'}** — {nsum.get('gemm_execs')}/{expect_execs} held-graph "
         f"executes, {len(blocks)} blocks run, fatal={len(fatal)}, "
         f"payload hash chain {'MATCH' if chain_match else ('N/A (real proc_a)' if chain_match is None else 'MISMATCH')}",
         f"- weights: one `ANeuralNetworksMemory_createFromFd` view of the artifact, per-GEMM "
         f"`setOperandValueFromMemory` windows (one stored INT8 copy); "
         f"compile {fu(ready.get('compile_ms'), '{:.0f}')} ms for the 9 graphs",
         "", "## Numbers (device, this run)", "",
         "| quantity | value |", "|---|---|",
         f"| 9-GEMM block (TPU), per block | **{fu(g9, '{:.1f}')} ms** "
         f"({fu(res['gemms9_gops'], '{:.0f}')} GOPS over {macs9/1e9:.3f} GMAC) |",
         f"| co-run reference for the same 9 GEMMs | 390.4 ms (`RESULTS_corun_tg.md`) ⇒ "
         f"{g9/390.4:.3f}× |",
         f"| per-token-tile GEMM table | see below |",
         f"| whole NPU block incl. down ({args.down}) | **{fu(bm.get('median'), '{:.1f}')} ms** median "
         f"(n={bm.get('n')}) |",
         f"| payload fp16→INT8 quantisation | {fu((res['quant_ms'] or {}).get('median'), '{:.1f}')} ms "
         f"median per chunk ({args.M}×{KD} = {args.M*KD:,} values) |",
         f"| publish→consume latency | {fu((res['publish_to_consume_us'] or {}).get('mean'), '{:.1f}')} µs mean, "
         f"{fu((res['publish_to_consume_us'] or {}).get('max'), '{:.1f}')} µs max |",
         f"| `ffn.down` on the TPU ({args.down}) | {down_probe_cell} |",
         f"| `ffn.down` on Mali (measured reference) | **{MALI_DOWN_MS:.2f} ms** per block "
         f"(`RESULTS_gemm_mali.md`) |",
         "", "### Per GEMM on the TPU (ms per block = 4 token tiles, this run)", "",
         "| GEMM | N | K | median ms | GOPS |", "|---|---|---|---|---|"]
    for n, s in pg.items():
        N = next((b.get("N") for b in blk_lines if b.get("gemm") == n), 0) or 0
        K = next((b.get("K") for b in blk_lines if b.get("gemm") == n), 0) or 0
        macs = 4 * 384 * N * K
        L.append(f"| `{n}` | {N} | {K} | {fu(s['median'], '{:.2f}')} | "
                 f"{fu(2.0 * macs / (s['median'] * 1e6), '{:.0f}')} |")
    down_tpu_cell = fu(per_block_down, "{:.2f}") if per_block_down else "not run this tick"
    down_ratio_cell = f"{(per_block_down / MALI_DOWN_MS):.2f}×" if per_block_down else "—"
    down_state_cell = ("measured above" if per_block_down
                       else "measured separately: 258.0 ms (`RESULTS_npu_tn.md` §4)")
    down_probe_cell = (fu(per_block_down, "{:.1f}") + " ms per block" if per_block_down
                       else "not executed (branch=skip)")
    L += ["", "`publish→consume` here is **queueing, not handoff cost**: the consumer is mid-block when "
          "the producer publishes, so this is how long the chunk waited. The primitive itself was "
          "measured free-running at **680.9 µs** mean in P3.1 (`RESULTS_handoff_p31.md`).", "",
          "## `ffn.down` — both branches, side by side", "",
          "| path | ms per block | vs Mali | state |", "|---|---|---|---|",
          f"| Mali-G78 tiled INT8 | **{MALI_DOWN_MS:.2f}** | 1.00× | measured (`RESULTS_gemm_mali.md`) |",
          f"| Edge TPU, 7×K=1280 + 6×ADD (gather + FC + ADD) | {down_tpu_cell} | "
          f"{down_ratio_cell} | {down_state_cell} |",
          f"| Edge TPU, single FC K=8960 | — | — | structurally impossible (compiler K ceiling 1664) |",
          "", "## Thermal", "",
          "- before: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_before.items())
                                  if k in ZONE_KEYS),
          "- after:  " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_after.items())
                                  if k in ZONE_KEYS),
          f"- gate (max zone ≤ battery+3): **{'PASS' if gate_ok else 'START OVER THE LINE'}** "
          f"— peak {fu(peak)} °C vs battery {fu(batt)} °C + 3 = {fu((batt or 0) + 3)} °C",
          "", "## What this proves / does not", "",
          "- proves: the NPU stage consumes the ring the GPU stage produces, runs the cut's "
          "`(180 - k)` blocks as **held** one-graph-per-GEMM compilations over the **real** INT8 "
          "artifact with zero weight copies, and releases each slot with a measured "
          "publish→consume latency",
          "- does not prove: numerical parity of the block output (the activation is the residual "
          "payload quantised at `SA = 1/127`; the real forward pass is Gate 3), that the TPU's "
          "per-tensor scales reproduce the locked per-column quantisation (measured cost: "
          "`RESULTS_scales.md`), or the steady-state period (P3.5)",
          "- the decode is a **stub**: `T_V` = 10.640 s (hybrid) is cited from "
          "`RESULTS_vae_tv.md`, not executed in this process",
          ]
    if warns:
        L += ["", "## Warnings", ""] + [f"- `{w}`" for w in warns[:10]]
    if fatal:
        L += ["", "## Fatal", ""] + [f"- `{f}`" for f in fatal[:10]]
    md = DEVICE_DIR / f"proc_b_{res['timestamp']}.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\nwrote {js.name} and {md.name}")
    print(f"  verdict={'PASS' if verdict else 'FAIL'} execs={nsum.get('gemm_execs')}/{expect_execs} "
          f"gemms9_ms={nsum.get('gemms9_ms_per_block')} block_median_ms={bm.get('median')} "
          f"chain={'MATCH' if chain_match else ('n/a' if chain_match is None else 'MISMATCH')}")
    for f in fatal[:5]:
        print("  " + f)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
