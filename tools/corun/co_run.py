#!/usr/bin/env python3
"""Co-run `T_G` — Mali-G78 (OpenCL) and Tensor Edge TPU (NNAPI) at the same time.

The cut needs one number that no solo measurement can supply: what a Preset-A
block costs on the GPU *while* the NPU is executing its own block, and vice
versa. Both engines read the same file-mapped INT8 artifact, so contention is a
real possibility and must be measured, not assumed.

    python tools/corun/co_run.py --build
    python tools/corun/co_run.py --solo-iters 10 --npu-rounds 60 --gpu-iters 25

Method
  * `gemm_bench block` times a whole Preset-A block (10 GEMMs) and now emits
    `TS|gpu|t0=..|t1=..` absolute CLOCK_MONOTONIC ms per iteration.
  * `npu_block corun` compiles the 9 compilable block GEMMs once, holds all of
    them, then loops rounds of `tiles` executes per GEMM, emitting `TS|npu|..`
    per execute. (Compiles are NOT interleaved with the timed work — they would
    contend with the GPU process on the CPU, not on the engine.)
  * The NPU process starts first and runs ~1.5 s alone, so the same process
    yields solo *and* contended samples: the comparison is an A/B on one code
    path at one thermal state, not a cross-capture contrast.
  * Overlap is computed from the timestamps themselves, so "they really ran at
    the same time" is evidence in the artifact, not a claim.

Nothing is modelled. A rate that was not measured stays null.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "probe"))
from probe_host import ADB, adb, pick_serial, read_zones, run  # noqa: E402

GEMM_DIR = REPO / "tools" / "android" / "gemm"
NPU_DIR = REPO / "tools" / "android" / "npublock"
CORUN = Path(__file__).resolve().parent
GEMM_BIN = GEMM_DIR / "build" / "gemm_bench"
NPU_BIN = NPU_DIR / "build" / "npu_block"
KERNEL = GEMM_DIR / "kernels" / "gemm8.cl"
DEVICE_SCRIPT = CORUN / "corun_device.sh"
DEVICE_DIR = REPO / "docs" / "device"

REMOTE = {
    GEMM_BIN: "/data/local/tmp/gemm_bench",
    NPU_BIN: "/data/local/tmp/npu_block",
    KERNEL: "/data/local/tmp/gemm8.cl",
    DEVICE_SCRIPT: "/data/local/tmp/corun_device.sh",
}
ZONE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")
# measured downsep ffn.down cost, docs/device/RESULTS_npu_tn.md §4 (…_174838)
DOWNSEP_S = 0.2580


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def kv(line: str, tag: str) -> dict:
    d = {}
    for part in line.split("|")[1:]:
        k, sep, v = part.partition("=")
        if sep:
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d


def samples(text: str, side: str) -> list[dict]:
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith(f"TS|{side}|"):
            continue
        d = kv(line, side)
        d["label"] = line.split("|")[2]
        out.append(d)
    return out


def overlaps(a0: float, a1: float, ivs: list[tuple[float, float]]) -> float:
    """Fraction of [a0,a1] covered by the union of intervals `ivs`."""
    if a1 <= a0 or not ivs:
        return 0.0
    cov = 0.0
    for b0, b1 in ivs:
        lo, hi = max(a0, b0), min(a1, b1)
        if hi > lo:
            cov += hi - lo
    return min(1.0, cov / (a1 - a0))


def stats(v: list[float]) -> dict | None:
    if not v:
        return None
    return {"n": len(v), "min": min(v), "median": st.median(v), "mean": sum(v) / len(v)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--solo-iters", type=int, default=10, help="GPU solo block samples")
    ap.add_argument("--npu-rounds", type=int, default=100, help="NPU rounds (9 GEMMs x tiles each)")
    ap.add_argument("--variant", default="tiled_64x64x32_t8x4", help="Mali GEMM variant name")
    ap.add_argument("--gpu-iters", type=int, default=25, help="GPU blocks during the co-run")
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--gpu-M", type=int, default=1530, help="tokens per Preset-A block (GPU)")
    ap.add_argument("--npu-M", type=int, default=384, help="per-graph token ceiling (NPU)")
    ap.add_argument("--cooldown", type=float, default=None,
                    help="gate on max(TPU,skin,VIRTUAL-SKIN) <= this C before the run")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--serial", default=None)
    args = ap.parse_args()

    if args.npu_M > 384:
        print("npu-M above the measured per-graph ceiling — refusing", file=sys.stderr)
        return 2
    if not ADB.exists():
        print(f"adb missing at {ADB}", file=sys.stderr)
        return 2
    if args.build:
        for d in (GEMM_DIR, NPU_DIR):
            print(f"building {d.name} ...")
            rc, txt = run(["bash", str(d / "build.sh").replace("\\", "/")], timeout=900)
            print(txt[-800:])
            if rc != 0:
                return rc
    for b in (GEMM_BIN, NPU_BIN):
        if not b.exists():
            print(f"{b.name} not built — rerun with --build", file=sys.stderr)
            return 2

    serial = args.serial or pick_serial()
    if not serial:
        print("no device", file=sys.stderr)
        return 2

    if args.cooldown is not None:
        zones = read_zones(serial)
        peak = max((v for k, v in zones.items() if k in ZONE_KEYS), default=None)
        print(f"thermal gate: peak={peak} C wants <= {args.cooldown} C")
        if peak is not None and peak > args.cooldown:
            print("TOO WARM — refusing a throttled run", file=sys.stderr)
            return 3

    # Windows write_file emits CRLF; the device shell (mksh/toybox) cannot parse
    # a script containing carriage returns. Normalise to LF before pushing.
    lf = Path(CORUN / "corun_device.lf.sh")
    raw = DEVICE_SCRIPT.read_bytes().replace(bytes([13, 10]), bytes([10]))
    lf.write_bytes(raw)
    push = dict(REMOTE)
    push[lf] = "/data/local/tmp/corun_device.sh"
    for local, remote in push.items():
        rc, txt = run([str(ADB), "-s", serial, "push", str(local), remote], timeout=180)
        if rc != 0:
            print(f"push failed {local}: {txt}", file=sys.stderr)
            return 2
    adb(serial, "shell chmod 755 /data/local/tmp/gemm_bench /data/local/tmp/npu_block "
                "/data/local/tmp/corun_device.sh")
    _r, _hdr = adb(serial, "shell head -c 40 /data/local/tmp/corun_device.sh | od -c | head -3")
    print(_hdr)
    adb(serial, "logcat -c")

    temps_before = read_zones(serial)
    cmd = (f"shell sh /data/local/tmp/corun_device.sh {args.solo_iters} "
           f"{args.npu_rounds} {args.gpu_iters} {args.tiles} {args.gpu_M} {args.npu_M} "
           f"{args.variant}")
    print(f"running: {cmd}  (timeout {args.timeout}s)")
    rc, out = adb(serial, cmd, timeout=args.timeout)
    temps_after = read_zones(serial)
    print(out[-2500:])

    texts = {}
    for name in ("co_gpu_solo.txt", "co_gpu.txt", "co_npu.txt"):
        r, t = adb(serial, f"shell cat /data/local/tmp/{name}", timeout=120)
        texts[name] = t if r == 0 else ""
        print(f"  {name}: {len(texts[name].splitlines())} lines")

    gpu_solo = samples(texts["co_gpu_solo.txt"], "gpu")
    gpu_co = samples(texts["co_gpu.txt"], "gpu")
    npu = samples(texts["co_npu.txt"], "npu")
    npu_rounds = [kv(l, "round") for l in texts["co_npu.txt"].splitlines()
                  if l.strip().startswith("CORUN_ROUND|")]
    npu_summary = [kv(l, "summary") for l in texts["co_npu.txt"].splitlines()
                   if l.strip().startswith("SUMMARY|")]
    npu_fatal = [l.strip() for l in texts["co_npu.txt"].splitlines() if l.strip().startswith("FATAL")]
    gpu_fatal = [l.strip() for l in texts["co_gpu.txt"].splitlines() if l.strip().startswith("FATAL")]

    gpu_ivs = [(g["t0"], g["t1"]) for g in gpu_co]
    npu_ivs = [(n["t0"], n["t1"]) for n in npu]

    # classify each sample by how much of it actually overlapped the other engine
    for g in gpu_co:
        g["ov"] = overlaps(g["t0"], g["t1"], npu_ivs)
    for g in gpu_solo:
        g["ov"] = 0.0
    for n in npu:
        n["ov"] = overlaps(n["t0"], n["t1"], gpu_ivs)

    # GPU samples are labelled with the LoRA mode the harness ran (0 = base block,
    # 1/2 = the detail-expert FP16 paths) — group by it so the base block can be
    # quoted on its own instead of averaged with the FP16 variants.
    def by_lora(samples_, pred=lambda g: True) -> dict:
        out: dict[int, list[float]] = {}
        for g in samples_:
            if pred(g):
                out.setdefault(int(g.get("lora", -1)), []).append(g["ms"])
        return {k: stats(v) for k, v in sorted(out.items())}

    gpu_groups = {"solo": by_lora(gpu_solo),
                  "corun_contended": by_lora(gpu_co, lambda g: g["ov"] >= 0.5),
                  "corun_uncontended": by_lora(gpu_co, lambda g: g["ov"] < 0.5)}
    gs = [g["ms"] for g in gpu_solo]
    gc_cont = [g["ms"] for g in gpu_co if g["ov"] >= 0.5]
    gc_free = [g["ms"] for g in gpu_co if g["ov"] < 0.5]
    np_cont = [n["ms"] for n in npu if n["ov"] >= 0.5]
    np_free = [n["ms"] for n in npu if n["ov"] < 0.5]

    per_gemm: dict[str, dict] = {}
    for n in npu:
        d = per_gemm.setdefault(n["label"], {"solo": [], "cont": []})
        (d["cont"] if n["ov"] >= 0.5 else d["solo"]).append(n["ms"])

    overlap_frac = 0.0
    if gpu_ivs and npu_ivs:
        lo = max(min(a for a, _ in gpu_ivs), min(a for a, _ in npu_ivs))
        hi = min(max(b for _, b in gpu_ivs), max(b for _, b in npu_ivs))
        if hi > lo:
            span = sum(b - a for a, b in gpu_ivs if b > lo and a < hi) or 1
            overlap_frac = sum(overlaps(a, b, npu_ivs) * (b - a) for a, b in gpu_ivs
                               if b > lo and a < hi) / span

    res = {
        "serial": serial,
        "timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "M_gpu": args.gpu_M, "M_npu": args.npu_M, "tiles": args.tiles, "solo_iters": args.solo_iters,
        "npu_rounds": args.npu_rounds, "gpu_iters": args.gpu_iters, "rc": rc,
        "binary_sha256": {"gemm_bench": sha256(GEMM_BIN), "npu_block": sha256(NPU_BIN),
                          "gemm8.cl": sha256(KERNEL)},
        "temps": {"before": temps_before, "after": temps_after},
        "gpu": {"solo": stats(gs), "corun_contended": stats(gc_cont),
                "corun_uncontended": stats(gc_free), "by_lora": gpu_groups,
                "samples": gpu_co},
        "variant": args.variant,
        "npu": {"solo": stats(np_free), "contended": stats(np_cont),
                "n_samples": len(npu), "rounds": npu_rounds, "summary": npu_summary,
                "fatal": npu_fatal},
        "per_gemm": {k: {"solo": stats(v["solo"]), "contended": stats(v["cont"])}
                     for k, v in per_gemm.items()},
        "overlap_fraction_of_gpu_time": overlap_frac,
        "gpu_fatal": gpu_fatal,
        "raw": texts,
    }

    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    ts = res["timestamp"]
    js = DEVICE_DIR / f"corun_tg_{ts}.json"
    js.write_text(json.dumps(res, indent=2), encoding="utf-8")

    def f(x, fmt="{:.3f}"):
        return "TBD" if x is None else fmt.format(x)

    L = [f"# Co-run T_G / T_N — {ts}", "",
         f"serial `{serial}` · M_gpu={args.gpu_M} M_npu={args.npu_M} tiles={args.tiles} · GPU solo iters={args.solo_iters} · "
         f"NPU rounds={args.npu_rounds} · GPU co-run iters={args.gpu_iters} · rc={rc}", "",
         "## Thermal", "",
         "- before: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_before.items()) if k in ZONE_KEYS),
         "- after:  " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_after.items()) if k in ZONE_KEYS),
         "", f"GPU-time overlap with NPU executes: **{overlap_frac * 100:.1f} %**", "",
         f"## GPU — Preset-A block (`{args.variant}`)", "",
         "| LoRA mode | condition | n | min ms | median ms | mean ms |", "|---|---|---|---|---|---|"]
    for cond, label in (("solo", "solo (same session)"),
                        ("corun_contended", "co-run, NPU executing"),
                        ("corun_uncontended", "co-run, NPU idle")):
        for lora, st_ in gpu_groups[cond].items():
            L.append(f"| {lora} | {label} | {st_['n']} | {f(st_['min'])} | "
                     f"{f(st_['median'])} | {f(st_['mean'])} |")
    L += ["", "## NPU — per GEMM (M=384 executes)", "",
          "| GEMM | n solo | solo mean ms | n contended | contended mean ms | delta |",
          "|---|---|---|---|---|---|"]
    for k, v in res["per_gemm"].items():
        s, c = v["solo"], v["contended"]
        d = "" if not (s and c) else f"{(c['mean'] / s['mean'] - 1) * 100:+.1f} %"
        L.append(f"| `{k}` | {f(s and s['n'], '{}')} | {f(s and s['mean'])} | "
                 f"{f(c and c['n'], '{}')} | {f(c and c['mean'])} | {d} |")
    L += ["", f"- NPU solo mean of all executes: {f(res['npu']['solo'] and res['npu']['solo']['mean'])} ms",
          f"- NPU contended mean of all executes: {f(res['npu']['contended'] and res['npu']['contended']['mean'])} ms",
          ""]
    if npu_summary:
        s = npu_summary[0]
        L.append(f"- `corun` summary (9 GEMMs, contended by construction of the loop): "
                 f"{s.get('per_block_s')} s/block, {s.get('gops')} GOPS")
    for fl in (gpu_fatal + npu_fatal):
        L.append(f"- `{fl}`")
    L.append("")
    md = DEVICE_DIR / f"corun_tg_{ts}.md"
    md.write_text("\n".join(L), encoding="utf-8")

    print("\n".join(L))
    print(f"\nwrote {js}\nwrote {md}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
