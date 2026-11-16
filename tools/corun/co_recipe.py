#!/usr/bin/env python3
"""P3.7d — co-run treatment of the *exact* NPU recipe (the cut may not use 0.7164 solo).

P3.7c measured the only bit-exact NPU stage on this HAL at 0.7164 s/block **solo**, while
the locked `T_N` (0.6484 s) and `T_G` (0.1474 s) are co-run numbers. So no cut argument can
use the exact recipe until the two engines have been timed together.

This driver runs `proc_b --recipe 1` (one graph per 384-column block, M=192 row tiles) and
`gemm_bench block` (the Preset-A GPU block) **at the same time**, both stamping
CLOCK_MONOTONIC, then classifies every NPU pass and every GPU block by how much of it actually
overlapped the other engine — the same method `co_run.py` used for the 9-graph stage:

  * GPU solo samples come first, in the same session (A/B on one code path, one thermal state).
  * The recipe is launched, runs ~`--delay` seconds alone (solo NPU samples), then the GPU joins.
  * `RECIPE|pass` lines carry t0/t1 (added in this commit), so contended vs solo is measured
    per GEMM per pass, not inferred.

    python tools/corun/co_recipe.py --build
    python tools/corun/co_recipe.py --reps 12 --gpu-iters 40 --delay 2.0
    python tools/corun/co_recipe.py --verify-only docs/device/corun_recipe_*.json

Nothing modelled: a rate that was not measured stays null.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "probe"))
from probe_host import ADB, adb, pick_serial, read_zones, run  # noqa: E402

GEMM_DIR = REPO / "tools" / "android" / "gemm"
PIPE_DIR = REPO / "tools" / "android" / "pipe"
CORUN = Path(__file__).resolve().parent
GEMM_BIN = GEMM_DIR / "build" / "gemm_bench"
PIPE_BIN = PIPE_DIR / "build" / "proc_b"
KERNEL = GEMM_DIR / "kernels" / "gemm8.cl"
DEVICE_SCRIPT = CORUN / "co_recipe_device.sh"
ART = REPO / "artifacts" / "blocks"
DEVICE_DIR = REPO / "docs" / "device"

REMOTE_GEMM = "/data/local/tmp/gemm_bench"
REMOTE_KERNEL = "/data/local/tmp/gemm8.cl"
REMOTE_PROC_B = "/data/local/tmp/hv_proc_b"
REMOTE_BLK = "/data/local/tmp/hv_blk.bin"
REMOTE_SCRIPT = "/data/local/tmp/co_recipe_device.sh"

ZONE_KEYS = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")
BLOCK_BYTES = 46_399_488

# cited references (never re-derived here)
REF_RECIPE_SOLO_MS = 716.4       # P3.7c, 4 runs 715.8/717.2/717.8/716.4
REF_RECIPE_SOLO_SPREAD = (715.8, 717.8)
REF_9GRAPH_CORUN_MS = 390.4      # RESULTS_corun_tg.md
REF_TG_CORUN_MS = 147.4          # RESULTS_corun_tg.md (gemm_bench block, lora 0)
DOWNSEP_S = 0.2580               # RESULTS_npu_tn.md §4 (ffn.down on TPU, solo)
T_V_HYBRID = 10.640              # RESULTS_vae_tv.md


def kv(line: str) -> dict:
    d: dict = {}
    for part in line.split("|")[1:]:
        k, sep, v = part.partition("=")
        if not sep:
            continue
        if k in ("name", "gemm", "act", "note", "tag"):
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
            "max": max(v)}


def overlap(a0: float, a1: float, ivs: list[tuple[float, float]]) -> float:
    if a1 <= a0 or not ivs:
        return 0.0
    cov = 0.0
    for b0, b1 in ivs:
        lo, hi = max(a0, b0), min(a1, b1)
        if hi > lo:
            cov += hi - lo
    return min(1.0, cov / (a1 - a0))


def gate_wait(serial: str, slack: float, max_wait: float, poll: float, confirm: int):
    """Wait for the thermal floor, then confirm on fresh samples.

    A failed confirm re-enters the wait loop instead of failing the gate: the TPU
    zone is whole-degree, so one sample can step up while the device has not moved
    (same defect fixed in tools/pipe/tn_recipe.py the same night — it aborted a
    floor-grade run at 27.0 PASS / 28.4 confirm).
    """
    t0 = time.time()
    while True:
        while True:
            z = read_zones(serial)
            peak = max((v for k, v in z.items() if k in ZONE_KEYS), default=None)
            batt = z.get("battery")
            if peak is None or batt is None:
                print("  [gate] zones unreadable — refusing a timed run")
                return False, z
            gate = batt + 3.0 + slack
            el = int(time.time() - t0)
            print(f"  [gate {el:4d}s] peak={peak:.1f} gate={gate:.1f}")
            if peak <= gate:
                break
            if time.time() - t0 > max_wait:
                print(f"  [gate] gave up after {el}s")
                return False, z
            time.sleep(poll)
        ok = True
        for i in range(max(1, confirm)):
            time.sleep(8)
            z = read_zones(serial)
            peak = max((v for k, v in z.items() if k in ZONE_KEYS), default=None)
            gate = (z.get("battery") or 99) + 3.0 + slack
            print(f"  [gate confirm {i + 1}/{max(1, confirm)}] peak={peak:.1f} gate={gate:.1f}")
            if peak is None or peak > gate:
                ok = False
                break
        if ok:
            return True, z
        if time.time() - t0 > max_wait:
            print("  [gate] confirm kept stepping over the gate; giving up")
            return False, z
        print("  [gate] confirm stepped over the gate — back to waiting")
        time.sleep(poll)


def analyse(gpu_solo_txt: str, gpu_co_txt: str, npu_txt: str) -> dict:
    gpu_solo = [kv(l) for l in gpu_solo_txt.splitlines() if l.strip().startswith("TS|gpu|")]
    gpu_co = [kv(l) for l in gpu_co_txt.splitlines() if l.strip().startswith("TS|gpu|")]
    cols = [kv(l) for l in npu_txt.splitlines() if l.strip().startswith("RECIPE|col|")]
    passes = [kv(l) for l in npu_txt.splitlines() if l.strip().startswith("RECIPE|pass|")]
    plan = next((kv(l) for l in npu_txt.splitlines() if l.strip().startswith("RECIPE|plan|")), {})
    summ = [kv(l) for l in npu_txt.splitlines() if l.strip().startswith("SUMMARY|recipe|")]
    fails = [l.strip() for l in npu_txt.splitlines()
             if l.strip().startswith(("FATAL", "RECIPE|FAIL", "WARN"))]

    gpu_ivs = [(g["t0"], g["t1"]) for g in gpu_co]
    # NPU busy intervals: union of col-samples and pass-samples (col spans the executes of one
    # graph over its row tiles, pass spans all column blocks of one GEMM) — using both keeps the
    # GPU-side overlap from being under-counted by the print gap between two samples.
    npu_ivs = [(c["t0"], c["t1"]) for c in cols if "t0" in c and "t1" in c]
    npu_ivs += [(p["t0"], p["t1"]) for p in passes if "t0" in p and "t1" in p]

    for g in gpu_co:
        g["ov"] = overlap(g["t0"], g["t1"], npu_ivs)
    for p in passes:
        p["ov"] = overlap(p["t0"], p["t1"], gpu_ivs) if "t0" in p else 0.0
    for c in cols:
        c["ov"] = overlap(c["t0"], c["t1"], gpu_ivs) if "t0" in c else 0.0

    def gpu_group(samples_, lora=0, pred=lambda x: True) -> dict | None:
        v = [g["ms"] for g in samples_ if int(g.get("lora", -1)) == lora and pred(g)]
        return stats(v)

    gpu = {
        "solo_lora0": gpu_group(gpu_solo, 0),
        "corun_lora0_contended": gpu_group(gpu_co, 0, lambda g: g["ov"] >= 0.5),
        "corun_lora0_uncontended": gpu_group(gpu_co, 0, lambda g: g["ov"] < 0.5),
        "by_lora": {str(l): {"solo": gpu_group(gpu_solo, l),
                             "contended": gpu_group(gpu_co, l, lambda g: g["ov"] >= 0.5),
                             "uncontended": gpu_group(gpu_co, l, lambda g: g["ov"] < 0.5)}
                    for l in (0, 1, 2)},
    }

    # per-GEMM per-pass totals, pass>=2 (pass 1 pays the cold weight fetch).
    # Samples are also split into "before the GPU joined" and "after it left", so the
    # contended comparison has a control at BOTH ends of the same hot/cool trajectory
    # instead of only at the start.
    per_gemm: dict[str, dict] = {}
    for p in passes:
        if int(p.get("pass", 0)) < 2:
            continue
        nm = str(p.get("name"))
        d = per_gemm.setdefault(nm, {"rows": []})
        d["rows"].append((int(p["pass"]), float(p["ms_tot"]), p["ov"]))

    gemm_rows = {}
    cont_total = 0.0
    solo_early_total = 0.0
    solo_late_total = 0.0
    cont_missing = []
    for nm, d in per_gemm.items():
        rows = sorted(d["rows"])
        cont_passes = [ps for ps, _ms, ov in rows if ov >= 0.5]
        first_c = min(cont_passes) if cont_passes else None
        last_c = max(cont_passes) if cont_passes else None
        early = [ms for ps, ms, ov in rows if first_c is not None and ps < first_c]
        late = [ms for ps, ms, ov in rows if last_c is not None and ps > last_c]
        cs = stats([ms for _ps, ms, ov in rows if ov >= 0.5])
        gemm_rows[nm] = {
            "solo_early": stats(early), "solo_late": stats(late),
            "contended": cs,
            "delta_vs_early_pct": (100.0 * (cs["median"] / st.median(early) - 1.0))
            if (cs and early) else None,
            "delta_vs_late_pct": (100.0 * (cs["median"] / st.median(late) - 1.0))
            if (cs and late) else None,
        }
        if cs:
            cont_total += cs["median"]
        else:
            cont_missing.append(nm)
        if early:
            solo_early_total += st.median(early)
        if late:
            solo_late_total += st.median(late)

    # block totals from the device's own SUMMARY lines (context; the pass-splits above are
    # the solo/contended decomposition the cut needs)
    block_ms = {int(s["pass"]): float(s["block_ms"]) for s in summ if "pass" in s}
    warm = [v for k, v in sorted(block_ms.items()) if k >= 2]

    gpu_ov = 0.0
    if gpu_ivs and npu_ivs:
        lo = max(min(a for a, _ in gpu_ivs), min(a for a, _ in npu_ivs))
        hi = min(max(b for _, b in gpu_ivs), max(b for _, b in npu_ivs))
        if hi > lo:
            span = sum(b - a for a, b in gpu_ivs if b > lo and a < hi) or 1
            gpu_ov = sum(overlap(a, b, npu_ivs) * (b - a) for a, b in gpu_ivs
                         if b > lo and a < hi) / span

    return {
        "plan": plan, "block_ms_per_pass": block_ms,
        "block_ms_warm_median": st.median(warm) if warm else None,
        "compile_ms": next((float(s["compile_ms"]) for s in summ if "compile_ms" in s), None),
        "gpu": gpu,
        "gpu_overlap_fraction": gpu_ov,
        "npu_per_gemm": gemm_rows,
        "npu_contended_total_ms": cont_total if not cont_missing else None,
        "npu_solo_early_total_ms": solo_early_total or None,
        "npu_solo_late_total_ms": solo_late_total or None,
        "npu_solo_session_total_ms":
            ((solo_early_total + solo_late_total) / 2.0) if (solo_early_total and solo_late_total)
            else (solo_early_total or solo_late_total or None),
        "npu_contended_missing": cont_missing,
        "npu_samples": {"cols": len(cols), "passes": len(passes),
                        "cols_contended": sum(1 for c in cols if c.get("ov", 0) >= 0.5)},
        "gpu_samples": {"solo": len(gpu_solo), "corun": len(gpu_co),
                        "contended": sum(1 for g in gpu_co if g["ov"] >= 0.5)},
        "fails": fails[:20],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--solo-iters", type=int, default=8)
    ap.add_argument("--gpu-iters", type=int, default=40)
    ap.add_argument("--reps", type=int, default=12, help="recipe passes over the 9 GEMMs")
    ap.add_argument("--gpu-M", type=int, default=1530)
    ap.add_argument("--variant", default="tiled_64x64x32_t8x4")
    ap.add_argument("--row", type=int, default=192)
    ap.add_argument("--col", type=int, default=384)
    ap.add_argument("--tokens", type=int, default=1536)
    ap.add_argument("--delay", type=float, default=2.0, help="s the recipe runs before the GPU joins")
    ap.add_argument("--block-idx", type=int, default=0)
    ap.add_argument("--gate-slack", type=float, default=1.0)
    ap.add_argument("--max-wait", type=float, default=1500)
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--confirm", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--verify-only", default=None, help="re-analyse an existing capture json")
    args = ap.parse_args()

    if args.verify_only:
        cap = json.loads(Path(args.verify_only).read_text(encoding="utf-8"))
        res = cap["analysis"]
        ts = cap["timestamp"]
        serial = cap.get("serial", "verify-only")
        temps_before = cap.get("temps", {}).get("before", {})
        temps_after = cap.get("temps", {}).get("after", {})
        gate_ok = cap.get("temps", {}).get("gate_ok", False)
    else:
        if args.build or not (GEMM_BIN.exists() and PIPE_BIN.exists()):
            for d in (GEMM_DIR, PIPE_DIR):
                print(f"building {d.name} ...")
                rc, txt = run(["bash", (d / "build.sh").as_posix()], timeout=900)
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
            peak = max((v for k, v in temps_before.items() if k in ZONE_KEYS), default=None)
            batt = temps_before.get("battery")
            print(f"  [gate] NO TIMED RUN: peak {peak} vs battery+3 "
                  f"{None if batt is None else batt + 3}")
            return 3

        # push (device shell cannot parse CRLF — normalise the script first)
        lf = CORUN / "co_recipe_device.lf.sh"
        lf.write_bytes(DEVICE_SCRIPT.read_bytes().replace(b"\r\n", b"\n"))
        pushes = [(GEMM_BIN, REMOTE_GEMM), (KERNEL, REMOTE_KERNEL),
                  (PIPE_BIN, REMOTE_PROC_B), (lf, REMOTE_SCRIPT)]
        for local, remote in pushes:
            rc, txt = run([str(ADB), "-s", serial, "push", str(local), remote], timeout=600)
            if rc != 0:
                print(f"push failed {local}: {txt}", file=sys.stderr)
                return 2
        adb(serial, f"shell chmod 755 {REMOTE_GEMM} {REMOTE_PROC_B} {REMOTE_SCRIPT}")
        _r, out = adb(serial, f"shell stat -c %s {REMOTE_BLK}")
        if out.strip() != str(BLOCK_BYTES):
            print(f"{REMOTE_BLK} is {out.strip()!r} — pushing the artifact block")
            rc, txt = run([str(ADB), "-s", serial, "push", str(blk), REMOTE_BLK], timeout=900)
            if rc != 0:
                print(f"push block failed: {txt}", file=sys.stderr)
                return 2
        adb(serial, "logcat -c")      # rule: clear the NNAPI buffer before any graph work

        cmd = (f"shell sh {REMOTE_SCRIPT} {args.solo_iters} {args.gpu_iters} {args.reps} "
               f"{args.gpu_M} {args.variant} {args.row} {args.col} {args.tokens} {args.delay}")
        print(f"  running: {cmd}  (timeout {args.timeout}s)")
        t0 = time.time()
        rc, raw = adb(serial, cmd, timeout=args.timeout)
        wall = time.time() - t0
        temps_after = read_zones(serial)
        print(f"  device wall {wall:.1f} s, rc={rc}")
        print(raw[-1500:])

        texts = {}
        for name in ("co_gpu_solo.txt", "co_gpu.txt", "co_recipe.txt"):
            r2, t2 = adb(serial, f"shell cat /data/local/tmp/{name}", timeout=180)
            texts[name] = t2 if r2 == 0 else ""
            print(f"  {name}: {len(texts[name].splitlines())} lines")
        res = analyse(texts["co_gpu_solo.txt"], texts["co_gpu.txt"], texts["co_recipe.txt"])
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------ derived rates
    g_solo = res["gpu"]["solo_lora0"]
    g_cont = res["gpu"]["corun_lora0_contended"]
    tn_cont_ms = res["npu_contended_total_ms"]
    tn_solo_ms = res["npu_solo_session_total_ms"]
    block_cont_ms = res["block_ms_warm_median"]

    cuts = {}
    if g_cont and tn_cont_ms:
        tg = g_cont["median"] / 1000.0
        tn_solo_s = (tn_solo_ms / 1000.0 + DOWNSEP_S) if tn_solo_ms else None
        tn = tn_cont_ms / 1000.0 + DOWNSEP_S
        for label, Tg, Tn in (("contended pair (this run)", tg, tn),
                              ("solo recipe (P3.7c) + locked T_G", REF_TG_CORUN_MS / 1000.0,
                               REF_RECIPE_SOLO_MS / 1000.0 + DOWNSEP_S)):
            cmd = [sys.executable, str(REPO / "phase2_profile" / "solve_cut.py"),
                   "--Tg", f"{Tg:.4f}", "--Tn", f"{Tn:.4f}", "--Tv", f"{T_V_HYBRID:.3f}"]
            p = subprocess.run(cmd, capture_output=True, text=True)
            cuts[label] = {"T_G": Tg, "T_N": Tn, "stdout": p.stdout.strip()}
        if tn_solo_s:
            cuts["this-run solo NPU + this-run T_G"] = {
                "T_G": tg, "T_N": tn_solo_s,
                "stdout": subprocess.run(
                    [sys.executable, str(REPO / "phase2_profile" / "solve_cut.py"),
                     "--Tg", f"{tg:.4f}", "--Tn", f"{tn_solo_s:.4f}", "--Tv",
                     f"{T_V_HYBRID:.3f}"], capture_output=True, text=True).stdout.strip()}

    cap = {"timestamp": ts, "serial": serial, "stage": "P3.7d co-run of the exact recipe",
           "plan": res["plan"], "temps": {"before": temps_before, "after": temps_after,
                                          "gate_ok": gate_ok},
           "analysis": res, "cuts": cuts,
           "references": {"recipe_solo_ms": REF_RECIPE_SOLO_MS,
                          "recipe_solo_spread": list(REF_RECIPE_SOLO_SPREAD),
                          "tg_corun_ms": REF_TG_CORUN_MS,
                          "graph9_corun_ms": REF_9GRAPH_CORUN_MS,
                          "downsep_s": DOWNSEP_S, "T_V_hybrid_s": T_V_HYBRID}}
    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    js = DEVICE_DIR / f"corun_recipe_{ts}.json"
    js.write_text(json.dumps(cap, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ markdown
    def f(x, fmt="{:.3f}"):
        return "TBD" if x is None else fmt.format(x)

    L = [f"# Co-run of the exact NPU recipe (P3.7d) — {ts}", "",
         f"serial `{serial}` · recipe row M={res['plan'].get('row')} col N={res['plan'].get('col')} "
         f"tokens={res['plan'].get('tokens')} reps={res['plan'].get('reps')} · GPU M={args.gpu_M} "
         f"`{args.variant}` · solo iters {args.solo_iters} · co-run iters {args.gpu_iters}", ""]
    if not res["npu_samples"]["passes"]:
        L.append("- **NO MEASUREMENT** — no `RECIPE|pass` lines came back (see raw log)")
    else:
        L += ["## Verdict", "",
              f"- NPU exact recipe under GPU load: **{f(tn_cont_ms, '{:.1f}')} ms/block** "
              f"(sum of per-GEMM medians, contended passes only) vs "
              f"**{f(res['npu_solo_early_total_ms'], '{:.1f}')} ms** before the GPU joined and "
              f"**{f(res['npu_solo_late_total_ms'], '{:.1f}')} ms** after it left — both inside this "
              f"run — and the P3.7c solo reference **{REF_RECIPE_SOLO_MS:.1f} ms**",
              f"- GPU base block while the recipe executes: "
              f"**{f(g_cont and g_cont['median'])} ms** vs **{f(g_solo and g_solo['median'])} ms** "
              f"solo in the same session (locked co-run reference {REF_TG_CORUN_MS:.1f} ms)",
              f"- engine-time overlap: {res['gpu_overlap_fraction'] * 100:.1f} % of GPU block "
              f"time fell inside NPU executes; {res['gpu_samples']['contended']}/"
              f"{res['gpu_samples']['corun']} GPU samples and "
              f"{res['npu_samples']['cols_contended']}/{res['npu_samples']['cols']} NPU "
              f"col-samples classified contended", ""]

    L += ["## GPU — Preset-A block (lora 0)", "",
          "| condition | n | min ms | median ms | mean ms |", "|---|---|---|---|---|"]
    for key, label in (("solo_lora0", "solo (same session)"),
                       ("corun_lora0_contended", "co-run, recipe executing"),
                       ("corun_lora0_uncontended", "co-run, NPU not executing")):
        s = res["gpu"][key]
        L.append(f"| {label} | {f(s and s['n'], '{}')} | {f(s and s['min'])} | "
                 f"{f(s and s['median'])} | {f(s and s['mean'])} |")
    L += ["", "## NPU exact recipe — per GEMM (pass ≥2, classified by overlap)", "",
          "| GEMM | solo before join | contended | Δ vs before | solo after GPU left "
          "| Δ vs after |", "|---|---|---|---|---|---|"]
    for nm, d in res["npu_per_gemm"].items():
        L.append(f"| `{nm}` | {f(d['solo_early'] and d['solo_early']['median'], '{:.2f}')} | "
                 f"{f(d['contended'] and d['contended']['median'], '{:.2f}')} | "
                 + (f"{d['delta_vs_early_pct']:+.1f} %" if d['delta_vs_early_pct'] is not None else "—")
                 + f" | {f(d['solo_late'] and d['solo_late']['median'], '{:.2f}')} | "
                 + (f"{d['delta_vs_late_pct']:+.1f} %" if d['delta_vs_late_pct'] is not None else "—")
                 + " |")
    L += ["",
          f"- contended total **{f(tn_cont_ms, '{:.1f}')} ms** (missing: "
          f"{res['npu_contended_missing'] or 'none'})",
          f"- device SUMMARY block (all passes, device-side): "
          f"{f(block_cont_ms, '{:.1f}')} ms median, pass 1 {f(res['block_ms_per_pass'].get(1), '{:.1f}')} ms",
          f"- compilations per block: 56, compile wall **{f(res['compile_ms'], '{:.0f}')} ms**", ""]

    if cuts:
        L += ["## Cut, re-solved on the measured pairs (T_V = 10.640 s hybrid)", ""]
        for label, c in cuts.items():
            L += [f"**{label}** — T_G {c['T_G']:.4f} s, T_N {c['T_N']:.4f} s", "", "```",
                  c["stdout"], "```", ""]

    L += ["## Thermal", "",
          "- before: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_before.items())
                                  if k in ZONE_KEYS),
          "- after:  " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_after.items())
                                  if k in ZONE_KEYS),
          f"- gate (`max zone <= battery+3`, slack {args.gate_slack:.1f} °C): "
          f"**{'PASS at start' if gate_ok else 'NOT GATED'}**", "",
          "## What this proves / does not", "",
          "- proves: the exact recipe's cost **while the GPU is executing the Preset-A block**, "
          "and the GPU's base-block cost while the exact recipe executes — both from one thermal "
          "state, with overlap computed from CLOCK_MONOTONIC stamps in two processes",
          "- does not prove: `ffn.down` under contention (the 0.2580 s downsep term is still "
          "solo), block-to-block variation, or that the graphs survive across chunk boundaries "
          "for the 8.64 s/block compile to amortise", ""]
    if res["fails"]:
        L += ["## Warnings / failures", ""] + [f"- `{x}`" for x in res["fails"][:10]]

    md = DEVICE_DIR / f"corun_recipe_{ts}.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    # raw logs for audit (gitignored like the other captures)
    if not args.verify_only:
        for name, txt in texts.items():
            (DEVICE_DIR / f"corun_recipe_{ts}.{name}.raw.log").write_text(txt, encoding="utf-8")

    print("\n".join(L[-40:]))
    print(f"\nwrote {md.name} and corun_recipe_{ts}.json")
    return 0 if res["npu_samples"]["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
