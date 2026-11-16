#!/usr/bin/env python3
"""P3.4 — the two-process chunk-pipeline coordinator (host side).

Launches the GPU stage (`tools/android/pipe/proc_a`, producer) and the NPU stage
(`tools/android/pipe/proc_b`, consumer) against ONE ring file on the phone, in one
`adb shell`, and measures what the cut actually needs:

  * per-chunk period  — GPU publish cadence and NPU finish cadence, from
    CLOCK_MONOTONIC stamps both processes print
  * publish->consume  — the ring latency the consumer observes
  * GPU / NPU busy    — summed compute intervals, plus the GPU's occupancy
    window (compute + payload conversion + publish)
  * overlap fraction  — union of the two engines' intervals, so "both engines
    busy at the same time" is a measurement rather than an assumption

    python phase3_pipeline/stream.py --chunks 4 --nblocks 1        # bounded proof
    python phase3_pipeline/stream.py --chunks 2 --nblocks 20       # the cut's real
                                                                   # 180-160 stage
    python phase3_pipeline/stream.py --chunks 4 --gate-wait 600    # cool first

Launch order is a correctness requirement, not a nicety: BOTH binaries call
`hv::reset_headers()` when they map the ring, so launching them together could
have the consumer wipe a publish the producer already made. The runner therefore
starts the consumer, waits for its `READY` line (compiled, ring mapped, headers
reset), and only then starts the producer.

Writes docs/device/stream_<ts>.{json,md,raw.log}. Every number in the doc comes
from this run's device output; anything not measured is written as TBD.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "probe"))
from probe_host import ADB, adb, pick_serial, read_zones, run  # noqa: E402

PIPE_DIR = REPO / "tools" / "android" / "pipe"
GEMM_DIR = REPO / "tools" / "android" / "gemm"
ART = REPO / "artifacts" / "blocks"
DEVICE_DIR = REPO / "docs" / "device"

R_A = "/data/local/tmp/hv_proc_a"
R_B = "/data/local/tmp/hv_proc_b"
R_CL = "/data/local/tmp/gemm8.cl"
R_BLK = "/data/local/tmp/hv_blk.bin"
R_SCL = "/data/local/tmp/hv_blk.scales.f32"
R_RING = "/data/local/tmp/hv_handoff.bin"
R_RUNNER = "/data/local/tmp/hv_stream.sh"
ZONE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")


def kv(line: str) -> dict:
    d: dict = {}
    for part in line.split("|")[1:]:
        k, sep, v = part.partition("=")
        if not sep:
            continue
        if k in ("name", "path", "gemm", "route", "src", "note", "branch"):
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


def diffs(v: list[float]) -> list[float]:
    return [b - a for a, b in zip(v, v[1:])]


def union_len(iv: list[tuple[float, float]]) -> float:
    if not iv:
        return 0.0
    s = sorted(iv)
    tot = 0.0
    cs, ce = s[0]
    for a, b in s[1:]:
        if a > ce:
            tot += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    return tot + (ce - cs)


def overlap_len(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Total time both engines were inside an interval: |A| + |B| - |A ∪ B|."""
    return union_len(a) + union_len(b) - union_len(a + b)


def remote_size(serial: str, path: str) -> int | None:
    rc, out = adb(serial, f"shell stat -c %s {path}")
    try:
        return int(out.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def gate(serial: str, wait_s: int) -> tuple[dict, float | None, float | None, bool]:
    """Spec gate: max(TPU, skin, VIRTUAL-SKIN) <= battery + 3 C."""
    t0 = time.time()
    while True:
        z = read_zones(serial)
        peak = max((v for k, v in z.items() if k in ZONE_KEYS), default=None)
        batt = z.get("battery")
        ok = (peak is not None and batt is not None and peak <= batt + 3.0)
        if ok or time.time() - t0 >= wait_s:
            return z, peak, batt, ok
        print(f"  gate: peak {peak:.1f} C vs {batt + 3:.1f} C — waiting "
              f"({int(time.time() - t0)}/{wait_s} s)")
        time.sleep(20)


def runner_text(serial: str, args, pub_every: int) -> str:
    b = (f"./hv_proc_b --block {R_BLK} --scales {R_SCL} --ring {R_RING} "
         f"--blocks {args.chunks} --nblocks {args.nblocks} --M {args.M} "
         f"--down {args.down} --source none --decode {args.decode} "
         f"--hash-every {args.hash_every}")
    a = (f"./hv_proc_a --block {R_BLK} --scales {R_SCL} --cl {R_CL} --ring {R_RING} "
         f"--blocks {args.chunks * pub_every} --pub-every {pub_every} "
         f"--once-blocks {args.once_blocks} --M {args.M} --route {args.route} --drain none "
         f"--warmup {args.warmup}")
    return (
        "#!/system/bin/sh\n"
        "cd /data/local/tmp || exit 2\n"
        "rm -f hv_a.log hv_b.log\n"
        ": > hv_a.log; : > hv_b.log\n"
        "# consumer first: it resets the ring headers, so nothing may be published yet\n"
        f"setsid {b} > hv_b.log 2>&1 < /dev/null &\n"
        "n=0\nwhile [ $n -lt 3000 ]; do grep -q '^READY' hv_b.log 2>/dev/null && break; "
        "sleep 0.1; n=$((n+1)); done\n"
        "echo \"STREAM|consumer_ready_polls=$n\"\n"
        f"setsid {a} > hv_a.log 2>&1 < /dev/null &\n"
        "n=0\nwhile [ $n -lt 3000 ]; do grep -q '^RESULT' hv_a.log 2>/dev/null && break; "
        "sleep 0.2; n=$((n+1)); done\n"
        "m=0\nwhile [ $m -lt 3000 ]; do grep -q '^RESULT' hv_b.log 2>/dev/null && break; "
        "sleep 0.2; m=$((m+1)); done\n"
        "sleep 0.3\n"
        "echo \"STREAM|gpu_polls=$n|npu_polls=$m\"\n"
        "echo 'MARK|A_BEGIN'; cat hv_a.log; echo 'MARK|A_END'\n"
        "echo 'MARK|B_BEGIN'; cat hv_b.log; echo 'MARK|B_END'\n"
    )


def split_logs(out: str) -> tuple[list[str], list[str]]:
    a: list[str] = []
    b: list[str] = []
    cur = 0
    for line in out.splitlines():
        s = line.strip()
        if s == "MARK|A_BEGIN":
            cur = 1
            continue
        if s == "MARK|A_END":
            cur = 0
            continue
        if s == "MARK|B_BEGIN":
            cur = 2
            continue
        if s == "MARK|B_END":
            cur = 0
            continue
        if cur == 1:
            a.append(s)
        elif cur == 2:
            b.append(s)
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--nblocks", type=int, default=1,
                    help="NPU blocks per chunk (the cut's 180 - k = 20). 1 keeps a unit test bounded.")
    ap.add_argument("--M", type=int, default=1530)
    ap.add_argument("--down", default="skip", choices=["skip", "stage"])
    ap.add_argument("--route", default="import", choices=["import", "copy"])
    ap.add_argument("--decode", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--block-idx", type=int, default=0)
    ap.add_argument("--gate-wait", type=int, default=0, help="seconds to wait for the thermal gate")
    ap.add_argument("--pub-every", type=int, default=None,
                    help="GPU blocks per handoff (default: --nblocks). The cut hands off once per "
                         "160 GPU blocks while the NPU runs 20 — that asymmetry is the pipeline.")
    ap.add_argument("--hash-every", type=int, default=1,
                    help="hash ffn.up's 13.7 MB output every N NPU blocks (1 = every block, the "
                         "P3.3/P3.4 behaviour; 20 removes a measured ~80 ms/block dead gap)")
    ap.add_argument("--once-blocks", type=int, default=4,
                    help="length of the GPU closure-reference pass (0/negative = all blocks)")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--serial", default=None)
    args = ap.parse_args()
    pub_every = args.pub_every if args.pub_every else args.nblocks
    if args.once_blocks <= 0:
        args.once_blocks = args.chunks * pub_every

    bins = {"a": PIPE_DIR / "build" / "proc_a", "b": PIPE_DIR / "build" / "proc_b"}
    if args.build or not all(p.exists() for p in bins.values()):
        print("building pipe binaries ...")
        rc, txt = run(["bash", str(PIPE_DIR / "build.sh")], timeout=900)
        if rc != 0:
            print(txt[-3000:], file=sys.stderr)
            raise SystemExit("build failed")
    blk = ART / f"block_{args.block_idx:02d}.bin"
    scl = ART / f"block_{args.block_idx:02d}.scales.f32"
    kernel = GEMM_DIR / "kernels" / "gemm8.cl"
    for p in (blk, scl, kernel, *bins.values()):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 2

    serial = args.serial or pick_serial()
    if not serial:
        print("no device attached", file=sys.stderr)
        return 2

    temps_before, peak, batt, gate_ok = gate(serial, args.gate_wait)
    print(f"  thermal: peak {peak} C, battery {batt} C, gate "
          f"{'PASS' if gate_ok else 'OVER THE LINE'}")

    for src, dst in ((bins["b"], R_B), (bins["a"], R_A), (kernel, R_CL)):
        rc, txt = run([str(ADB), "-s", serial, "push", str(src), dst], timeout=600)
        if rc != 0:
            print(f"push failed {src}: {txt}", file=sys.stderr)
            return 2
    adb(serial, f"shell chmod 755 {R_B}")
    adb(serial, f"shell chmod 755 {R_A}")
    if remote_size(serial, R_BLK) != blk.stat().st_size:
        rc, txt = run([str(ADB), "-s", serial, "push", str(blk), R_BLK], timeout=900)
        if rc != 0:
            print(f"push block failed: {txt}", file=sys.stderr)
            return 2
    if remote_size(serial, R_SCL) != scl.stat().st_size:
        run([str(ADB), "-s", serial, "push", str(scl), R_SCL], timeout=600)

    adb(serial, "logcat -c")          # rule: clear the NNAPI buffer before graph work

    tmp = Path(tempfile.gettempdir()) / "hv_stream_runner.sh"
    tmp.write_bytes(runner_text(serial, args, pub_every).encode())
    rc, txt = run([str(ADB), "-s", serial, "push", str(tmp), R_RUNNER], timeout=300)
    if rc != 0:
        print(f"push runner failed: {txt}", file=sys.stderr)
        return 2

    print(f"  running: chunks={args.chunks} nblocks={args.nblocks} pub-every={pub_every} "
          f"gpu-blocks={args.chunks * pub_every} hash-every={args.hash_every} M={args.M} "
          f"down={args.down} route={args.route}")
    t_wall0 = time.time()
    rc, out = adb(serial, f"shell sh {R_RUNNER}", timeout=args.timeout)
    wall_s = time.time() - t_wall0
    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    (DEVICE_DIR / f"stream_{ts}.raw.log").write_text(out, encoding="utf-8")

    a_lines, b_lines = split_logs(out)
    stream = [kv(l) for l in out.splitlines() if l.startswith("STREAM|")]
    fatal = [l for l in a_lines + b_lines if l.startswith("FATAL")]
    warns = [l for l in a_lines + b_lines if l.startswith("WARN") or l.startswith("EXEC_ERR")]
    temps_after = read_zones(serial)

    # ------------------------------------------------------------------ GPU side
    gpu_ts = [kv(l) for l in a_lines if l.startswith("TS|gpu|")]
    gpu_pubts = [g for g in gpu_ts if g.get("pub") == 1]
    gpu_blocks = [g for g in gpu_ts if isinstance(g.get("block_ms"), (int, float))]
    gpu_ms = [float(g["block_ms"]) for g in gpu_blocks]
    gpu_once = [float(kv(l)["ms_once"]) for l in a_lines
                if l.startswith("TSC|gpu|") and "ms_once" in l]
    gpu_occ: list[tuple[float, float]] = []       # compute + payload + publish
    gpu_iv: list[tuple[float, float]] = []        # compute only
    pub_ns = [int(g["t_pub_ns"]) for g in gpu_pubts if "t_pub_ns" in g]
    for g in gpu_blocks:
        t0 = float(g["t0_ms"])
        t1 = float(g["t1_ms"])
        gpu_iv.append((t0, t1))
        end = float(g["t_pub_ns"]) / 1e6 if g.get("pub") == 1 and "t_pub_ns" in g else t1
        gpu_occ.append((t0, end))
    gpu_a = {l.split("|")[1]: kv(l) for l in a_lines if l.startswith("SUMMARY|")}
    gpu_result = next((kv(l) for l in a_lines if l.startswith("RESULT|")), {})
    gpu_wtot = next((kv(l) for l in a_lines if l.startswith("W_TOTAL|")), {})
    gpu_close = next((kv(l) for l in a_lines if l.startswith("CLOSE|")), {})
    slot_wait_us = [float(g["slot_wait_us"]) for g in gpu_blocks if "slot_wait_us" in g]
    payload_ms = [float(g["payload_ms"]) for g in gpu_blocks if "payload_ms" in g]

    # ------------------------------------------------------------------ NPU side
    npu_ts = [kv(l) for l in b_lines if l.startswith("TS|npu|")]
    npu_blocks = [t for t in npu_ts if "block" in t and isinstance(t.get("ms"), (int, float))]
    npu_cons = [t for t in npu_ts if "slot" in t]
    npu_block_ms = [float(t["ms"]) for t in npu_blocks]
    lat_us = [float(t["lat_ns"]) / 1e3 for t in npu_cons if isinstance(t.get("lat_ns"), (int, float))]
    quant_ms = [float(t["quant_ms"]) for t in npu_cons if isinstance(t.get("quant_ms"), (int, float))]
    nan_words = sum(int(t.get("nan_words", 0)) for t in npu_cons)
    zero_words = sum(int(t.get("zero_words", 0)) for t in npu_cons)
    npu_ready = next((kv(l) for l in b_lines if l.startswith("READY|")), {})
    npu_a = {l.split("|")[1]: kv(l) for l in b_lines if l.startswith("SUMMARY|")}
    npu_result = next((kv(l) for l in b_lines if l.startswith("RESULT|")), {})
    npu_thr = next((kv(l) for l in b_lines if l.startswith("THROUGHPUT|")), {})

    # NPU per-chunk occupancy: from the consumed-publish stamp to the last block end.
    npu_iv: list[tuple[float, float]] = []
    npu_chunk_ms: list[float] = []
    npu_chunk_end: list[float] = []
    for t in npu_cons:
        ch = int(t["chunk"])
        blks = [x for x in npu_blocks if int(x.get("chunk", -1)) == ch]
        if not blks:
            continue
        t_start = None
        lat = float(t.get("lat_ns", 0))
        if pub_ns and ch - 1 < len(pub_ns):
            t_start = (pub_ns[ch - 1] + int(lat)) / 1e6     # t_prod + observed latency
        t_end = max(float(x["t1"]) for x in blks)
        t_first = min(float(x["t0"]) for x in blks)
        start = t_start if t_start is not None else t_first
        npu_iv.append((start, t_end))
        npu_chunk_ms.append(t_end - start)
        npu_chunk_end.append(t_end)

    gpu_chunk_end = [float(g["t_pub_ns"]) / 1e6 for g in gpu_blocks if "t_pub_ns" in g]
    ov = overlap_len(gpu_occ, npu_iv)
    gpu_busy = union_len(gpu_occ)
    npu_busy = union_len(npu_iv)
    span = (max([e for _, e in gpu_occ + npu_iv]) - min([s for s, _ in gpu_occ + npu_iv])) \
        if (gpu_occ and npu_iv) else 0.0

    chunks = args.chunks

    def ratio(x) -> tuple[int, int] | None:
        """'4/4' -> (4, 4). The device prints ratios, so parity must be parsed, not compared."""
        if isinstance(x, str) and "/" in x:
            a, _, b = x.partition("/")
            try:
                return int(a), int(b)
            except ValueError:
                return None
        if isinstance(x, int):
            return x, x
        return None

    gpu_pub = ratio(gpu_result.get("published"))
    npu_done = ratio(npu_result.get("chunks"))
    npu_execs = ratio(npu_result.get("gemm_execs"))
    verdict = (rc == 0 and not fatal
               and len(pub_ns) == chunks
               and len(gpu_blocks) == chunks * pub_every
               and len(npu_cons) == chunks
               and nan_words == 0
               and gpu_pub == (chunks, chunks)
               and npu_done == (chunks, chunks)
               and len(npu_iv) == chunks)

    res = {
        "serial": serial,
        "timestamp": ts,
        "stage": "P3.4/P3.5 stream.py (coordinator: proc_a producer + proc_b consumer, one ring)",
        "M": args.M, "chunks": chunks, "nblocks": args.nblocks,
        "pub_every": pub_every, "gpu_blocks_run": len(gpu_blocks),
        "hash_every": args.hash_every,
        "down": args.down, "route": args.route, "warmup": args.warmup,
        "wall_s": wall_s,
        "launch": stream,
        "gpu": {
            "chunks_published": len(pub_ns),
            "blocks_run": len(gpu_blocks),
            "blocks_published": len(pub_ns),
            "block_ms": stats(gpu_ms),
            "block_ms_series": gpu_ms,
            "block_ms_once": stats(gpu_once),
            "payload_ms": stats(payload_ms),
            "slot_wait_us": stats(slot_wait_us),
            "closure_us": gpu_close.get("overhead_us"),
            "weights": gpu_wtot,
            "publish_interval_s": stats([d / 1e9 for d in diffs([float(x) for x in pub_ns])]),
            "publish_offsets_s": [(p - pub_ns[0]) / 1e9 for p in pub_ns] if pub_ns else [],
            "summary": gpu_a, "result": gpu_result,
        },
        "npu": {
            "chunks_consumed": len(npu_cons),
            "block_ms": stats(npu_block_ms),
            "block_ms_series": npu_block_ms,
            "chunk_window_ms": stats(npu_chunk_ms),
            "chunk_window_series_ms": npu_chunk_ms,
            "finish_interval_s": stats([d / 1e3 for d in diffs(npu_chunk_end)]),
            "publish_to_consume_us": stats(lat_us),
            "quant_ms": stats(quant_ms),
            "gemm_execs": npu_execs,
            "ready": npu_ready,
            "zero_words": zero_words, "nan_words": nan_words,
            "summary": npu_a, "throughput": npu_thr, "result": npu_result,
        },
        "overlap": {
            "gpu_busy_ms": gpu_busy, "npu_busy_ms": npu_busy,
            "union_ms": union_len(gpu_occ + npu_iv),
            "both_busy_ms": ov,
            "span_ms": span,
            "fraction_of_npu_busy": (ov / npu_busy) if npu_busy else None,
            "fraction_of_gpu_busy": (ov / gpu_busy) if gpu_busy else None,
            "fraction_of_span": (ov / span) if span else None,
            "method": "interval union of CLOCK_MONOTONIC windows: GPU = block t0..t_pub_ns "
                      "(compute + payload + publish), NPU = (t_prod + observed latency)..last block t1",
        },
        "temps": {"before": temps_before, "after": temps_after,
                  "peak_before_c": peak, "battery_before_c": batt, "gate_ok": gate_ok,
                  "gate": "max(TPU, skin1, skin2, VIRTUAL-SKIN) <= battery + 3 C"},
        "fatal": fatal[:10], "warnings": warns[:20], "rc": rc,
        "verdict": bool(verdict),
    }
    js = DEVICE_DIR / f"stream_{ts}.json"
    js.write_text(json.dumps(res, indent=2), encoding="utf-8")

    def fu(x, fmt="{:.1f}"):
        return "TBD" if x is None else fmt.format(x)

    gb, nb = res["gpu"]["block_ms"] or {}, res["npu"]["block_ms"] or {}
    ovl = res["overlap"]
    L = [f"# P3.4/P3.5 — coordinator: two stages, one ring — {ts}", "",
         f"serial `{serial}` · M={args.M} · **{chunks} chunks × {args.nblocks} NPU block(s)** · "
         f"GPU **{args.chunks * pub_every} blocks, handoff every {pub_every}** · "
         f"GPU route **{args.route}** · `ffn.down` = {args.down} · "
         f"`ffn.up` hash every {args.hash_every} NPU block(s) · wall **{wall_s:.1f} s**", "",
         "## Verdict", "",
         f"- **{'PASS' if verdict else 'FAIL'}** — GPU published **{len(pub_ns)}/{chunks}** chunks over "
         f"{len(gpu_blocks)} blocks, "
         f"NPU consumed **{len(npu_cons)}/{chunks}**, fatal={len(fatal)}, "
         f"NaN words={nan_words}, zero words={zero_words}, rc={rc}",
         f"- weights: {res['gpu']['weights'].get('imported', '?')}/"
         f"{res['gpu']['weights'].get('gems', '?')} imported zero-copy on the GPU side, "
         f"one `createFromFd` view on the NPU side (one stored INT8 copy)",
         "", "## Numbers (device, this run)", "",
         "| quantity | value |", "|---|---|",
         f"| GPU block (all {len(gpu_blocks)} blocks) | **{fu(gb.get('median'))} ms** median "
         f"(min {fu(gb.get('min'))}, max {fu(gb.get('max'))}, n={gb.get('n')}) |",
         f"| NPU block on the TPU | **{fu(nb.get('median'))} ms** median (n={nb.get('n')}) |",
         f"| `ffn.up` output hash cost | "
         f"{fu((npu_a.get('npu') or {}).get('out_hash_ms_total'), '{:.1f}')} ms over "
         f"{(npu_a.get('npu') or {}).get('out_hash_calls', '?')} hash(es) |",
         f"| chunk window, NPU | {fu((res['npu']['chunk_window_ms'] or {}).get('median'))} ms median |",
         f"| GPU publish interval | {fu((res['gpu']['publish_interval_s'] or {}).get('median'), '{:.3f}')} s "
         f"median between chunk publishes |",
         f"| NPU chunk finish interval | "
         f"{fu((res['npu']['finish_interval_s'] or {}).get('median'), '{:.3f}')} s median |",
         f"| publish→consume | {fu((res['npu']['publish_to_consume_us'] or {}).get('mean'), '{:.2f}')} µs mean, "
         f"{fu((res['npu']['publish_to_consume_us'] or {}).get('max'), '{:.2f}')} µs max |",
         f"| GPU back-pressure (slot wait) | {fu((res['gpu']['slot_wait_us'] or {}).get('max'), '{:.0f}')} µs max |",
         f"| payload conversion (host-side) | {fu((res['gpu']['payload_ms'] or {}).get('median'), '{:.1f}')} ms per chunk |",
         "", "## Overlap (both engines executing at once)", "",
         "| quantity | value |", "|---|---|",
         f"| GPU busy (occupancy windows) | {gpu_busy:.1f} ms |",
         f"| NPU busy | {npu_busy:.1f} ms |",
         f"| both busy | **{ov:.1f} ms** |",
         f"| of NPU busy | **{fu((ovl['fraction_of_npu_busy'] or 0) * 100, '{:.1f}')} %** |",
         f"| of GPU busy | {fu((ovl['fraction_of_gpu_busy'] or 0) * 100, '{:.1f}')} % |",
         f"| of the measured span ({span:.1f} ms) | {fu((ovl['fraction_of_span'] or 0) * 100, '{:.1f}')} % |",
         "", "## Thermal", "",
         "- before: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_before.items())
                                 if k in ZONE_KEYS),
         "- after:  " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_after.items())
                                 if k in ZONE_KEYS),
         f"- gate (max zone ≤ battery+3): **{'PASS' if gate_ok else 'START OVER THE LINE'}** — "
         f"peak {fu(peak)} °C vs battery {fu(batt)} °C + 3 = {fu((batt or 0) + 3)} °C",
         "", "## What this proves / does not", "",
         "- proves: one `adb shell` brings both stages up in the correct order (consumer maps and "
         "resets the ring first, then the producer publishes into it), the GPU stage publishes "
         "every chunk, the NPU stage consumes every chunk and releases the slot, and the overlap "
         "between the two engines is measured from their own CLOCK_MONOTONIC stamps",
         "- does not prove: numerical parity (Gate 3), the steady-state period at the cut's real "
         "cadence (P3.5), or the decode — which is still a cited stub, not an execution",
         ]
    if warns:
        L += ["", "## Warnings", ""] + [f"- `{w}`" for w in warns[:8]]
    if fatal:
        L += ["", "## Fatal", ""] + [f"- `{f}`" for f in fatal[:8]]
    md = DEVICE_DIR / f"stream_{ts}.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\nwrote {js.name} and {md.name}")
    print(f"  verdict={'PASS' if verdict else 'FAIL'} gpu={len(gpu_blocks)}/{chunks} "
          f"npu={len(npu_cons)}/{chunks} gpu_block_med={gb.get('median')} "
          f"npu_block_med={nb.get('median')} overlap={ov:.1f}ms "
          f"({fu((ovl['fraction_of_npu_busy'] or 0) * 100, '{:.1f}')}% of NPU busy)")
    for f in fatal[:5]:
        print("  " + f)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
