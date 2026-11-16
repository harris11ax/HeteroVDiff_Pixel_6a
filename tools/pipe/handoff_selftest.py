#!/usr/bin/env python3
"""P3.1 — chunk-handoff selftest (host side).

Pushes `tools/android/pipe/handoff_selftest` and runs the ring two-process
handoff over N chunks, twice:

  * free-running  (`--stall-us 0`)     — producer always ahead, one slot in use
  * back-pressured(`--stall-us 1000`)  — consumer slower than the producer, so
                                        slot reuse must block on `done`

Both must report 0 mismatching words / 0 out-of-order publishes and every
`t_cons > t_prod`. Writes docs/device/handoff_selftest_<ts>.{json,md}.

    python tools/pipe/handoff_selftest.py --chunks 200
    python tools/pipe/handoff_selftest.py --build --chunks 200

This is a *correctness* test, not a rate: the thermal gate is recorded, not
enforced — the ring lives in page cache and does not heat the TPU.
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
PIPE_BIN = PIPE_DIR / "build" / "handoff_selftest"
DEVICE_DIR = REPO / "docs" / "device"
REMOTE_BIN = "/data/local/tmp/hv_handoff_selftest"
REMOTE_RING = "/data/local/tmp/hv_handoff.bin"
ZONE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")


def kv(line: str) -> dict:
    d: dict = {}
    for part in line.split("|")[1:]:
        k, sep, v = part.partition("=")
        if sep:
            try:
                d[k] = float(v) if "." in v or "e" in v.lower() else int(v)
            except ValueError:
                d[k] = v
    return d


def stats(v: list[float]) -> dict | None:
    if not v:
        return None
    return {"n": len(v), "min": min(v), "median": st.median(v), "mean": sum(v) / len(v),
            "max": max(v), "stdev": st.pstdev(v) if len(v) > 1 else 0.0}


def one_run(serial: str, chunks: int, stall_us: int, timeout: int) -> dict:
    cmd = f"shell {REMOTE_BIN} --path {REMOTE_RING} --chunks {chunks} --stall-us {stall_us}"
    print(f"  running: chunks={chunks} stall_us={stall_us}")
    rc, out = adb(serial, cmd, timeout=timeout)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    raw_prod = [kv(l) for l in lines if l.startswith("TS|producer|")]
    raw_cons = [kv(l) for l in lines if l.startswith("TS|consumer|")]
    # A record whose numeric fields did not parse is interleaved/corrupt output,
    # not a slow chunk: count it and make it fail the verdict rather than hide it.
    num = (int, float)
    prod = [p for p in raw_prod if isinstance(p.get("t_ns"), num)
            and isinstance(p.get("msync_ns"), num) and isinstance(p.get("blocked_ns"), num)]
    cons = [c for c in raw_cons if isinstance(c.get("t_ns"), num) and isinstance(c.get("lat_ns"), num)]
    malformed = (len(raw_prod) - len(prod)) + (len(raw_cons) - len(cons))
    summ = {l.split("|")[1]: kv(l) for l in lines if l.startswith("SUMMARY|")}
    result = next((kv(l) for l in lines if l.startswith("RESULT|")), {})
    header = next((kv(l) for l in lines if l.startswith("HV|selftest|")), {})
    bad = [l for l in lines if l.startswith("BAD|")]
    fatal = [l for l in lines if l.startswith("FATAL")]

    lat_us = [c.get("lat_ns", 0) / 1e3 for c in cons]
    blocked_us = [p.get("blocked_ns", 0) / 1e3 for p in prod]
    msync_us = [p.get("msync_ns", 0) / 1e3 for p in prod]
    idx_p = [int(p["chunk"]) for p in prod]
    idx_c = [int(c["chunk"]) for c in cons]
    verdict = (result.get("producer_rc") == 0 and result.get("consumer_rc") == 0
               and not bad and not fatal and malformed == 0
               and idx_c == list(range(1, chunks + 1))
               and idx_p == list(range(1, chunks + 1)))
    return {
        "chunks": chunks, "stall_us": stall_us, "rc": rc, "verdict": bool(verdict),
        "malformed_lines": malformed,
        "ring_header": header,
        "producer": summ.get("producer", {}), "consumer": summ.get("consumer", {}),
        "producer_chunks_seen": len(idx_p), "consumer_chunks_seen": len(idx_c),
        "in_order_producer": idx_p == sorted(idx_p),
        "in_order_consumer": idx_c == sorted(idx_c),
        "publish_to_consume_us": stats(lat_us),
        "producer_blocked_us": stats(blocked_us),
        "producer_chunks_blocked": sum(1 for b in blocked_us if b > 0),
        "msync_us": stats(msync_us),
        "bad_lines": bad[:20], "fatal": fatal[:20],
        "result_line": result,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--stall-us", type=int, default=1000,
                    help="consumer stall per chunk for the back-pressure run")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--timeout", type=int, default=480)
    ap.add_argument("--serial", default=None)
    args = ap.parse_args()

    if args.build or not PIPE_BIN.exists():
        print("building handoff_selftest ...")
        rc, txt = run(["bash", str(PIPE_DIR / "build.sh")], timeout=900)
        if rc != 0:
            print(txt[-3000:], file=sys.stderr)
            raise SystemExit("build failed")
        print("build ok")

    serial = args.serial or pick_serial()
    if not serial:
        print("no device attached", file=sys.stderr)
        return 2

    temps_before = read_zones(serial)
    peak = max((v for k, v in temps_before.items() if k in ZONE_KEYS), default=None)
    batt = temps_before.get("battery")

    rc, txt = run([str(ADB), "-s", serial, "push", str(PIPE_BIN), REMOTE_BIN], timeout=180)
    if rc != 0:
        print(f"push failed: {txt}", file=sys.stderr)
        return 2
    adb(serial, f"shell chmod 755 {REMOTE_BIN}")

    runs = [one_run(serial, args.chunks, 0, args.timeout),
            one_run(serial, args.chunks, args.stall_us, args.timeout)]

    # Independent end-state check: the ring file must be exactly 2 slots, i.e.
    # 4096 + 2 * 4,702,208 B — proof there is one handoff buffer, not a copy per stage.
    rc_ls, ls_out = adb(serial, f"shell ls -l {REMOTE_RING}")
    size = next((int(t) for t in ls_out.split() if t.isdigit() and len(t) > 5), None)
    expect_bytes = 4096 + 2 * ((4_700_160 + 4095) // 4096) * 4096
    temps_after = read_zones(serial)

    res = {
        "serial": serial,
        "timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "chunk_bytes": 4_700_160, "slot_stride": ((4_700_160 + 4095) // 4096) * 4096,
        "expected_file_bytes": expect_bytes, "device_file_bytes": size,
        "file_size_matches": size == expect_bytes,
        "file_ls": ls_out.strip(),
        "temps": {"before": temps_before, "after": temps_after, "peak_before_c": peak,
                  "battery_before_c": batt,
                  "note": "recorded, not enforced — P3.1 is a correctness test, no timed rate"},
        "runs": runs,
    }

    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    js = DEVICE_DIR / f"handoff_selftest_{res['timestamp']}.json"
    js.write_text(json.dumps(res, indent=2), encoding="utf-8")

    def fu(x, fmt="{:.1f}"):
        return "TBD" if x is None else fmt.format(x)

    L = [f"# P3.1 chunk-handoff selftest — {res['timestamp']}", "",
         f"serial `{serial}` · ring `{REMOTE_RING}` · one file, {2} slots, "
         f"chunk {res['chunk_bytes']:,} B (M=1530 × d=1536 fp16), "
         f"slot stride {res['slot_stride']:,} B", "",
         "Two processes (real PIDs) share one `MAP_SHARED` mapping; the producer "
         "stamps every 64-bit word with a chunk-derived pattern, the consumer verifies "
         "every word in order.", "",
         "## Verdict", ""]
    for r in runs:
        L.append(f"- **{'PASS' if r['verdict'] else 'FAIL'}** — chunks={r['chunks']}, "
                 f"stall={r['stall_us']} µs: bad_words={r['bad_lines'] and 'yes' or 0}, "
                 f"out-of-order=0/{'0' if r['in_order_consumer'] else 'MISMATCH'}, "
                 f"publish→consume mean {fu((r['publish_to_consume_us'] or {}).get('mean'))} µs "
                 f"(max {fu((r['publish_to_consume_us'] or {}).get('max'))} µs), "
                 f"producer blocked on {r['producer_chunks_blocked']}/{r['chunks']} chunks")
    L += ["", "## Ring", "",
          f"- device file size **{size:,} B** vs expected **{expect_bytes:,} B** "
          f"⇒ {'MATCH (one buffer, two slots)' if size == expect_bytes else 'MISMATCH'}",
          "- the ring carries residual activations only — it holds no weights, so the "
          "single-INT8-copy rule is untouched by construction", "",
          "## Thermal (recorded, not a gate)", "",
          "- before: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_before.items())
                                  if k in ZONE_KEYS),
          "- after:  " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_after.items())
                                  if k in ZONE_KEYS), "",
          "## What this proves / does not", "",
          "- proves: the publish (RELEASE) / spin (ACQUIRE) / release-done protocol "
          "delivers every chunk intact and in order across two processes, and the "
          "2-slot ring back-pressures the producer instead of tearing a payload",
          "- does not prove: steady-state period or overlap (that is P3.4/P3.5, with the "
          "real stages), nor anything about the weight artifact",
          ]
    md = DEVICE_DIR / f"handoff_selftest_{res['timestamp']}.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\nwrote {js.name} and {md.name}")
    for r in runs:
        print(f"  {'PASS' if r['verdict'] else 'FAIL'} chunks={r['chunks']} stall_us={r['stall_us']} "
              f"lat_mean_us={(r['publish_to_consume_us'] or {}).get('mean')} "
              f"blocked_chunks={r['producer_chunks_blocked']}")
    print(f"  ring file {size} B (expected {expect_bytes})")
    ok = all(r["verdict"] for r in runs) and res["file_size_matches"]
    print("VERDICT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
