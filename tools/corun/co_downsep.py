#!/usr/bin/env python3
"""P3.7j — `ffn.down`'s TPU branch (downsep) under GPU load: the last solo term in T_N.

Every other rate the cut uses is a co-run number. The one exception is `ffn.down` on the
TPU — 7 x FC[1536,1280] + 6 x ADD, `RESULTS_npu_tn.md` §4's 0.2580 s/block, measured alone.
So T_N's exact form (recipe + downsep) still carries a solo term, and the pipeline runs
`--down skip` precisely because nobody had contended it.

This driver runs, in one `adb shell` on one boot:

  A  GPU Preset-A blocks alone                     (solo T_G, same session)
  B  `proc_b --recipe 1`, a few passes alone       (solo recipe block, *this* boot —
                                                    P3.7h showed rates are boot-bounded)
  C  `npu_block downsep` alone                     (solo downsep tiles, contending control)
  D  `npu_block downsep` in the background while
     `gemm_bench block` executes                   (contended downsep tiles + contended T_G)

Every downsep tile and every GPU block stamps CLOCK_MONOTONIC, so each tile is classified
"ran while the GPU was busy" vs "ran alone" from the artifact, not from the arm.

    python tools/corun/co_downsep.py --build
    python tools/corun/co_downsep.py --down-solo-iters 12 --down-iters 40
    python tools/corun/co_downsep.py --verify-only docs/device/corun_downsep_*.json

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_host import ADB, adb, pick_serial, read_zones, run  # noqa: E402
from co_recipe import gate_wait, kv, overlap, stats, ZONE_KEYS  # noqa: E402

GEMM_DIR = REPO / "tools" / "android" / "gemm"
NPU_DIR = REPO / "tools" / "android" / "npublock"
PIPE_DIR = REPO / "tools" / "android" / "pipe"
CORUN = Path(__file__).resolve().parent
GEMM_BIN = GEMM_DIR / "build" / "gemm_bench"
NPU_BIN = NPU_DIR / "build" / "npu_block"
PIPE_BIN = PIPE_DIR / "build" / "proc_b"
KERNEL = GEMM_DIR / "kernels" / "gemm8.cl"
DEVICE_SCRIPT = CORUN / "co_downsep_device.sh"
ART = REPO / "artifacts" / "blocks"
DEVICE_DIR = REPO / "docs" / "device"

REMOTE_GEMM = "/data/local/tmp/gemm_bench"
REMOTE_KERNEL = "/data/local/tmp/gemm8.cl"
REMOTE_NPU = "/data/local/tmp/npu_block"
REMOTE_PROC_B = "/data/local/tmp/hv_proc_b"
REMOTE_BLK = "/data/local/tmp/hv_blk.bin"
REMOTE_SCRIPT = "/data/local/tmp/co_downsep_device.sh"

BLOCK_BYTES = 46_399_488

# cited references (never re-derived here)
REF_DOWNSEP_SOLO_S = 0.2580      # RESULTS_npu_tn.md §4 (7 x K=1280 + 6 ADD, TPU, solo)
REF_RECIPE_POSTBOOT_MS = 639.44  # RESULTS_tn_recipe.md §P3.7h, start TPU 25.0
REF_RECIPE_POSTBOOT_HI_MS = 651.23  # §P3.7i, start TPU 31.0
REF_RECIPE_CORUN_MS = 635.8      # RESULTS_corun_recipe.md §P3.7g, contended
REF_TG_SOLO_MS = 161.8           # RESULTS_gemm_mali.md
T_V_HYBRID = 10.640              # RESULTS_vae_tv.md
KNOWN_BOOT = "2026-11-16 02:22:27"   # the boot that splits every exact-recipe capture


def boot_time(serial: str) -> dict:
    rc, out = adb(serial, "shell cat /proc/uptime")
    try:
        up = float(out.split()[0])
    except (ValueError, IndexError):
        return {"uptime_s": None, "boot": None}
    b = dt.datetime.now() - dt.timedelta(seconds=up)
    return {"uptime_s": up, "boot": b.strftime("%Y-%m-%d %H:%M:%S"),
            "boot_epoch": b.timestamp()}


def analyse(gpu_solo_txt: str, gpu_co_txt: str, npu_solo_txt: str, npu_co_txt: str,
            recipe_txt: str) -> dict:
    gpu_solo = [kv(l) for l in gpu_solo_txt.splitlines() if l.strip().startswith("TS|gpu|")]
    gpu_co = [kv(l) for l in gpu_co_txt.splitlines() if l.strip().startswith("TS|gpu|")]
    t_solo = [kv(l) for l in npu_solo_txt.splitlines() if l.strip().startswith("TS|downsep|")]
    t_co = [kv(l) for l in npu_co_txt.splitlines() if l.strip().startswith("TS|downsep|")]
    recipe_summ = [kv(l) for l in recipe_txt.splitlines()
                   if l.strip().startswith("SUMMARY|recipe|")]
    fails = [l.strip() for l in (npu_solo_txt + npu_co_txt + recipe_txt).splitlines()
             if l.strip().startswith(("FATAL", "WARN"))]

    gpu_ivs = [(g["t0"], g["t1"]) for g in gpu_co if "t0" in g and "t1" in g]
    for t in t_co:
        t["ov"] = overlap(t["t0"], t["t1"], gpu_ivs)
    for g in gpu_co:
        npu_ivs = [(t["t0"], t["t1"]) for t in t_co]
        g["ov"] = overlap(g["t0"], g["t1"], npu_ivs)

    def g(v):
        return stats([float(x["ms"]) for x in v])

    cont = [t for t in t_co if t["ov"] >= 0.5]
    alone_in_corun = [t for t in t_co if t["ov"] < 0.5]
    recipe_warm = [float(s["block_ms"]) for s in recipe_summ
                   if "pass" in s and int(s["pass"]) >= 2]

    gpu_solo_l0 = [float(x["ms"]) for x in gpu_solo if int(x.get("lora", -1)) == 0]
    gpu_co_l0_rows = [x for x in gpu_co if int(x.get("lora", -1)) == 0]
    gpu_co_l0 = [float(x["ms"]) for x in gpu_co_l0_rows]
    gpu_co_cont = [float(x["ms"]) for x in gpu_co_l0_rows if x.get("ov", 0) >= 0.5]

    return {
        "downsep_solo_tiles": g(t_solo),
        "downsep_corun_alone_tiles": g(alone_in_corun),
        "downsep_corun_contended_tiles": g(cont),
        "downsep_tiles": {"solo": len(t_solo), "corun": len(t_co), "contended": len(cont)},
        "downsep_delta_pct": (100.0 * (st.median([float(t["ms"]) for t in cont])
                                       / st.median([float(t["ms"]) for t in t_solo]) - 1.0))
        if (cont and t_solo) else None,
        "downsep_first_vs_last_pct_contended":
            (100.0 * (float(cont[-1]["ms"]) / float(cont[0]["ms"]) - 1.0)) if len(cont) > 1 else None,
        "recipe_block_ms_warm": stats(recipe_warm),
        "gpu": {"solo_lora0": stats(gpu_solo_l0), "corun_lora0": stats(gpu_co_l0),
                "corun_lora0_contended": stats(gpu_co_cont)},
        "gpu_overlap": (st.median([x["ov"] for x in gpu_co_l0_rows]) if gpu_co_l0_rows else None),
        "fails": fails[:20],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--solo-iters", type=int, default=10, help="GPU blocks alone")
    ap.add_argument("--gpu-iters", type=int, default=90, help="GPU blocks in the co-run")
    ap.add_argument("--down-solo-iters", type=int, default=12, help="downsep iters alone")
    ap.add_argument("--down-iters", type=int, default=48, help="downsep iters in the co-run")
    ap.add_argument("--tiles", type=int, default=4, help="384-token tiles per block")
    ap.add_argument("--down-M", type=int, default=384)
    ap.add_argument("--gpu-M", type=int, default=1530)
    ap.add_argument("--variant", default="tiled_64x64x32_t8x4")
    ap.add_argument("--recipe-ref", type=int, default=8,
                    help="solo recipe passes in this session (0 = skip)")
    ap.add_argument("--delay", type=float, default=6.0,
                    help="s before the GPU joins (compile+first tiles land in it)")
    ap.add_argument("--block-idx", type=int, default=0)
    ap.add_argument("--gate-slack", type=float, default=0.0)
    ap.add_argument("--max-wait", type=float, default=1500)
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--confirm", type=int, default=2)
    ap.add_argument("--allow-over", action="store_true",
                    help="run even if the gate never opens, labelled floor-grade")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--verify-only", default=None)
    ap.add_argument("--from-logs", default=None,
                    help="TS of a saved capture: re-analyse docs/device/corun_downsep_TS.*.raw.log "
                         "+ .ctx.json without touching the device")
    args = ap.parse_args()

    texts: dict[str, str] = {}
    if args.from_logs:
        # Re-analysis path: must short-circuit BEFORE the thermal gate and the device
        # work, or re-parsing a saved capture starts a second device run (2026-11-16).
        tag = args.from_logs
        names = ("co_down_gpu_solo.txt", "co_down_gpu.txt", "co_down_npu_solo.txt",
                 "co_down_npu.txt", "co_down_recipe.txt")
        texts = {}
        for n in names:
            p = DEVICE_DIR / f"corun_downsep_{tag}.{n}.raw.log"
            texts[n] = p.read_text(encoding="utf-8") if p.exists() else ""
        ctxp = DEVICE_DIR / f"corun_downsep_{tag}.ctx.json"
        ctx = json.loads(ctxp.read_text(encoding="utf-8")) if ctxp.exists() else {}
        if not ctx:
            print(f"missing {ctxp.name} — re-parse only a capture that has one", file=sys.stderr)
            return 2
        res = analyse(texts["co_down_gpu_solo.txt"], texts["co_down_gpu.txt"],
                      texts["co_down_npu_solo.txt"], texts["co_down_npu.txt"],
                      texts["co_down_recipe.txt"])
        return emit(res, texts, tag, ctx.get("serial", "?"), ctx.get("temps_before", {}),
                    ctx.get("temps_after", {}), ctx.get("gate_ok", False),
                    ctx.get("boot", {}), ctx.get("meta", {}))
    if args.verify_only:
        cap = json.loads(Path(args.verify_only).read_text(encoding="utf-8"))
        res = cap["analysis"]
        ts = cap["timestamp"]
        serial = cap.get("serial", "verify-only")
        temps_before = cap.get("temps", {}).get("before", {})
        temps_after = cap.get("temps", {}).get("after", {})
        gate_ok = cap.get("temps", {}).get("gate_ok", False)
        boot = cap.get("boot", {})
    else:
        if args.build or not (GEMM_BIN.exists() and NPU_BIN.exists() and PIPE_BIN.exists()):
            for d in (GEMM_DIR, NPU_DIR, PIPE_DIR):
                print(f"building {d.name} ...")
                rc, txt = run(["bash", (d / "build.sh").as_posix()], timeout=900)
                if rc != 0:
                    print(txt[-3000:], file=sys.stderr)
                    return 2
        serial = args.serial or pick_serial()
        if not serial:
            print("no device attached", file=sys.stderr)
            return 2

        print("  gating the run thermally ...")
        gate_ok, temps_before = gate_wait(serial, args.gate_slack, args.max_wait,
                                          args.poll, args.confirm)
        if not gate_ok:
            peak = max((v for k, v in temps_before.items() if k in ZONE_KEYS), default=None)
            batt = temps_before.get("battery")
            gate = None if batt is None else batt + 3.0
            print(f"  [gate] peak {peak} vs battery+3 {gate}")
            if not args.allow_over:
                print("  [gate] NO TIMED RUN (pass --allow-over to run floor-grade)")
                return 3
            print("  [gate] running anyway — FLOOR-GRADE, labelled in the artifact")

        boot = boot_time(serial)

        lf = CORUN / "co_downsep_device.lf.sh"
        lf.write_bytes(DEVICE_SCRIPT.read_bytes().replace(b"\r\n", b"\n"))
        pushes = [(GEMM_BIN, REMOTE_GEMM), (KERNEL, REMOTE_KERNEL),
                  (NPU_BIN, REMOTE_NPU), (lf, REMOTE_SCRIPT)]
        if args.recipe_ref > 0:
            pushes.append((PIPE_BIN, REMOTE_PROC_B))
        for local, remote in pushes:
            rc, txt = run([str(ADB), "-s", serial, "push", str(local), remote], timeout=600)
            if rc != 0:
                print(f"push failed {local}: {txt}", file=sys.stderr)
                return 2
        adb(serial, f"shell chmod 755 {REMOTE_GEMM} {REMOTE_NPU} {REMOTE_SCRIPT}"
                    + (f" {REMOTE_PROC_B}" if args.recipe_ref > 0 else ""))
        if args.recipe_ref > 0:
            _r, out = adb(serial, f"shell stat -c %s {REMOTE_BLK}")
            if out.strip() != str(BLOCK_BYTES):
                print(f"{REMOTE_BLK} is {out.strip()!r} — pushing the artifact block")
                rc, txt = run([str(ADB), "-s", serial, "push",
                               str(ART / f"block_{args.block_idx:02d}.bin"), REMOTE_BLK],
                              timeout=900)
                if rc != 0:
                    print(f"push block failed: {txt}", file=sys.stderr)
                    return 2
        adb(serial, "logcat -c")     # rule: clear the NNAPI buffer before any graph work

        cmd = (f"shell sh {REMOTE_SCRIPT} {args.solo_iters} {args.gpu_iters} "
               f"{args.down_solo_iters} {args.down_iters} {args.gpu_M} {args.variant} "
               f"{args.tiles} {args.down_M} {args.delay} {args.recipe_ref}")
        print(f"  running: {cmd}  (timeout {args.timeout}s)")
        t0 = time.time()
        rc, raw = adb(serial, cmd, timeout=args.timeout)
        wall = time.time() - t0
        temps_after = read_zones(serial)
        print(f"  device wall {wall:.1f} s, rc={rc}")
        print(raw[-1200:])

        for name in ("co_down_gpu_solo.txt", "co_down_gpu.txt", "co_down_npu_solo.txt",
                     "co_down_npu.txt", "co_down_recipe.txt"):
            r2, t2 = adb(serial, f"shell cat /data/local/tmp/{name}", timeout=180)
            texts[name] = t2 if r2 == 0 else ""
            print(f"  {name}: {len(texts[name].splitlines())} lines")
        res = analyse(texts["co_down_gpu_solo.txt"], texts["co_down_gpu.txt"],
                      texts["co_down_npu_solo.txt"], texts["co_down_npu.txt"],
                      texts["co_down_recipe.txt"])
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        # The device visit is the expensive, one-chance part: persist the raw logs AND the
        # session context before analysing, so an analysis bug costs a re-parse, not a re-run
        # (a crash after a 125 s run + 4 min gate on 2026-11-16 is exactly that lesson).
        meta = {"tiles": args.tiles, "down_M": args.down_M, "gpu_M": args.gpu_M,
                "variant": args.variant, "gate_slack": args.gate_slack,
                "down_iters": args.down_iters, "down_solo_iters": args.down_solo_iters,
                "recipe_ref": args.recipe_ref, "delay": args.delay}
        for name, txt in texts.items():
            (DEVICE_DIR / f"corun_downsep_{ts}.{name}.raw.log").write_text(txt, encoding="utf-8")
        (DEVICE_DIR / f"corun_downsep_{ts}.ctx.json").write_text(json.dumps(
            {"timestamp": ts, "serial": serial, "boot": boot, "meta": meta,
             "temps_before": temps_before, "temps_after": temps_after,
             "gate_ok": gate_ok}, indent=2), encoding="utf-8")

    if args.from_logs:
        return 2   # unreachable: handled above, before any device work
    return emit(res, texts, ts, serial, temps_before, temps_after, gate_ok, boot, meta)


def emit(res: dict, texts: dict, ts: str, serial: str, temps_before: dict,
         temps_after: dict, gate_ok: bool, boot: dict, meta: dict) -> int:
    """Capture JSON + report from an analysis dict. Idempotent: re-parsing overwrites."""
    tiles = int(meta.get("tiles", 4))
    solo_tile = (res["downsep_solo_tiles"] or {}).get("median")
    cont_tile = (res["downsep_corun_contended_tiles"] or {}).get("median")
    down_solo_s = (solo_tile * tiles / 1000.0) if solo_tile else None
    down_cont_s = (cont_tile * tiles / 1000.0) if cont_tile else None
    recipe_s = ((res["recipe_block_ms_warm"] or {}).get("median", 0) / 1000.0) or None
    g_solo = res["gpu"]["solo_lora0"]
    g_cont = res["gpu"]["corun_lora0_contended"] or res["gpu"]["corun_lora0"]
    tg = (g_cont or {}).get("median", 0) / 1000.0 or None

    tn = None
    if recipe_s and down_cont_s:
        tn = recipe_s + down_cont_s
    cuts = {}
    if tn and tg:
        for label, Tn in ((f"this session, contended downsep ({tn:.4f} s)", tn),
                          (f"this session, solo downsep ({down_solo_s:.4f} s)",
                           recipe_s + down_solo_s) if down_solo_s else None):
            if Tn is None:
                continue
            p = subprocess.run([sys.executable, str(REPO / "phase2_profile" / "solve_cut.py"),
                                "--Tg", f"{tg:.4f}", "--Tn", f"{Tn:.4f}",
                                "--Tv", f"{T_V_HYBRID:.3f}"],
                               capture_output=True, text=True)
            cuts[label] = {"T_G": tg, "T_N": Tn, "stdout": p.stdout.strip()}

    cap = {"timestamp": ts, "serial": serial, "stage": "P3.7j ffn.down (downsep) under contention",
           "boot": boot, "plan": {"tiles": tiles, "down_M": meta.get("down_M"),
                                  "gpu_M": meta.get("gpu_M"), "variant": meta.get("variant"),
                                  "down_iters": meta.get("down_iters"),
                                  "down_solo_iters": meta.get("down_solo_iters"),
                                  "recipe_ref": meta.get("recipe_ref"),
                                  "delay_s": meta.get("delay")},
           "temps": {"before": temps_before, "after": temps_after, "gate_ok": gate_ok},
           "analysis": res, "cuts": cuts,
           "derived": {"downsep_solo_s_per_block": down_solo_s,
                       "downsep_contended_s_per_block": down_cont_s,
                       "recipe_s_per_block": recipe_s, "T_G_s": tg, "T_N_exact_s": tn},
           "references": {"downsep_solo_s": REF_DOWNSEP_SOLO_S,
                          "recipe_postboot_ms": REF_RECIPE_POSTBOOT_MS,
                          "recipe_postboot_hi_ms": REF_RECIPE_POSTBOOT_HI_MS,
                          "recipe_corun_ms": REF_RECIPE_CORUN_MS,
                          "tg_solo_ms": REF_TG_SOLO_MS, "T_V_hybrid_s": T_V_HYBRID,
                          "known_boot": KNOWN_BOOT}}
    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    js = DEVICE_DIR / f"corun_downsep_{ts}.json"
    js.write_text(json.dumps(cap, indent=2), encoding="utf-8")

    def f(x, fmt="{:.3f}"):
        return "TBD" if x is None else fmt.format(x)

    L = [f"# `ffn.down` (downsep) under contention (P3.7j) — {ts}", "",
         f"serial `{serial}` · downsep tiles={tiles} × M={meta.get('down_M')} "
         f"(one 1536-token block) · GPU M={meta.get('gpu_M')} `{meta.get('variant')}` · "
         f"boot `{boot.get('boot')}`", ""]
    if not res["downsep_tiles"]["contended"]:
        L.append("- **NO MEASUREMENT** — no contended downsep tiles came back (see raw log)")
    else:
        L += ["## Verdict", "",
              f"- `ffn.down` TPU path alone: **{f(solo_tile, '{:.3f}')} ms/tile** "
              f"(n={res['downsep_tiles']['solo']}) ⇒ **{f(down_solo_s)} s/block** "
              f"(reference 0.2580 s/block, `RESULTS_npu_tn.md` §4)",
              f"- while the Mali block executes: **{f(cont_tile, '{:.3f}')} ms/tile** "
              f"(n={res['downsep_tiles']['contended']}) ⇒ **{f(down_cont_s)} s/block** — "
              f"**{f(res['downsep_delta_pct'], '{:+.2f}')} % vs solo**",
              f"- same-session solo recipe block: **{f((res['recipe_block_ms_warm'] or {}).get('median'), '{:.2f}')} ms** "
              f"(n={(res['recipe_block_ms_warm'] or {}).get('n')}) ⇒ `T_N` exact = "
              f"**{f(tn)} s** with the contended downsep",
              f"- GPU Preset-A block contended **{f((g_cont or {}).get('median'), '{:.2f}')} ms** vs "
              f"solo **{f((g_solo or {}).get('median'), '{:.2f}')} ms**; median GPU-side overlap "
              f"**{(res['gpu_overlap'] or 0) * 100:.1f} %**", ""]
        L += ["| arm | n | min ms/tile | median ms/tile | max ms/tile |", "|---|---|---|---|---|"]
        for key, label in (("downsep_solo_tiles", "downsep alone"),
                           ("downsep_corun_alone_tiles", "co-run arm, GPU not executing"),
                           ("downsep_corun_contended_tiles", "co-run arm, GPU executing")):
            s = res[key]
            L.append(f"| {label} | {f(s and s['n'], '{}')} | {f(s and s['min'])} | "
                     f"{f(s and s['median'])} | {f(s and s['max'])} |")
        L += [""]
    if cuts:
        L += ["## Cut, re-solved (T_V = 10.640 s hybrid)", ""]
        for label, c in cuts.items():
            L += [f"**{label}** — T_G {c['T_G']:.4f} s, T_N {c['T_N']:.4f} s", "", "```",
                  c["stdout"], "```", ""]
    L += ["## Thermal", "",
          "- before: " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_before.items())
                                  if k in ZONE_KEYS),
          "- after:  " + " ".join(f"{k}={v:.1f}" for k, v in sorted(temps_after.items())
                                  if k in ZONE_KEYS),
          f"- gate (`max zone <= battery+3`, slack {float(meta.get('gate_slack') or 0):.1f} °C): "
          f"**{'PASS at start' if gate_ok else 'FLOOR-GRADE (gate did not open)'}**", "",
          "## What this proves / does not", "",
          "- proves: the TPU `ffn.down` path's cost **while the GPU executes a Preset-A block**, "
          "classified per 384-token tile from CLOCK_MONOTONIC stamps in two processes, against "
          "a solo control in the same session and boot",
          "- does not prove: the downsep against real `ffn.up` activations (the mode's activation "
          "is the deterministic fill the 0.2580 reference used — latency-only, no numerics "
          "claimed), per-block variation, or the recipe's own contended number on this boot", ""]
    if res["fails"]:
        L += ["## Warnings / failures", ""] + [f"- `{x}`" for x in res["fails"][:10]]

    md = DEVICE_DIR / f"corun_downsep_{ts}.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {md.name} and corun_downsep_{ts}.json")
    return 0 if res["downsep_tiles"]["contended"] else 1


if __name__ == "__main__":
    sys.exit(main())
