#!/usr/bin/env python3
"""Cut solver — paper Eq. 4/5 for the flattened trajectory.

    period(k_b) = max(k_b * T_G, (B - k_b) * T_N + T_V)
    k*          = (B * T_N + T_V) / (T_N + T_G)

`k*` is only the continuous crossing point; the deployed cut is the integer
`k_b in [1, B-1]` with the smallest period (paper §4.3). Everything here is
pure arithmetic on measured rates, so it can be tested without the phone.

Usage
    python phase2_profile/solve_cut.py --rates configs/rates_preset_a.json
    python phase2_profile/solve_cut.py --Tg 0.9 --Tn 0.35 --Tv 22 --B 180 --json out.json
    python phase2_profile/solve_cut.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def period(k_b: int, B: int, TG: float, TN: float, TV: float) -> float:
    return max(k_b * TG, (B - k_b) * TN + TV)


def solve(B: int, TG: float, TN: float, TV: float) -> dict:
    """Return the continuous crossing, the integer optimum, and the full sweep."""
    if TG <= 0 or TN <= 0:
        raise ValueError("T_G and T_N must be positive")
    k_cont = (B * TN + TV) / (TN + TG)
    valid = [k for k in range(1, B)]
    sweep = [(k, period(k, B, TG, TN, TV)) for k in valid]
    k_best, p_best = min(sweep, key=lambda kv: (kv[1], kv[0]))

    # neighbours of the continuous crossing, per the paper ("the neighbouring
    # integer that gives the smaller period")
    lo = max(1, min(B - 1, int(k_cont)))
    hi = max(1, min(B - 1, lo + 1))
    p_lo, p_hi = period(lo, B, TG, TN, TV), period(hi, B, TG, TN, TV)
    k_neighbour = lo if p_lo <= p_hi else hi

    # step-level alternatives (multiples of L) and their regret
    L = 30
    steps = [(k, period(k, B, TG, TN, TV)) for k in range(L, B, L)]
    k_step, p_step = min(steps, key=lambda kv: kv[1]) if steps else (None, float("inf"))

    # the two ablations the paper reports
    k_no_tv = min(valid, key=lambda k: period(k, B, TG, TN, 0.0))
    p_no_tv = period(k_no_tv, B, TG, TN, TV)

    return {
        "B": B, "T_G": TG, "T_N": TN, "T_V": TV,
        "k_continuous": k_cont,
        "k_b_opt": k_best,
        "period_at_opt": p_best,
        "k_b_from_neighbours": k_neighbour,
        "period_max_plus": max(p_lo, p_hi),
        "k_b_step_level": k_step,
        "period_step_level": p_step,
        "step_level_regret_pct": 100.0 * (p_step / p_best - 1.0) if p_step else None,
        "k_b_ignoring_TV": k_no_tv,
        "period_ignoring_TV": p_no_tv,
        "ignore_TV_regret_pct": 100.0 * (p_no_tv / p_best - 1.0),
        "gpu_stage_s": k_best * TG,
        "npu_stage_s": (B - k_best) * TN + TV,
        "idle_s": abs(k_best * TG - ((B - k_best) * TN + TV)),
        "speedup_vs_npu_only": ((B * TN + TV) / p_best) if p_best else None,
        "notes": _boundary_note(k_cont, B),
        "sweep_top10": sorted(sweep, key=lambda kv: kv[1])[:10],
    }


def _boundary_note(k_cont: float, B: int) -> str:
    if k_cont >= B:
        return ("crossing at or beyond B: even with zero GPU blocks the NPU stage cannot keep up; "
                "the integer search clamps to k_b = B-1 (GPU takes everything but one block)")
    if k_cont <= 1:
        return "crossing at or below 1: the GPU is slower than the NPU per block; prefix is 1 block"
    return "interior crossing"


def selftest() -> int:
    ok = True
    # 1. ANCHOR: the paper's own Table 5 row for the deployed cut implies
    #    T_G = 676/51 = 13.25 s, T_N = (675-267)/129 = 3.16 s, T_V = 267 s.
    #    The solver must return their k_b = 51 and their period of 683 s.
    r = solve(180, 13.2549, 3.1628, 267.0)
    print(f"paper anchor  : k*={r['k_continuous']:.1f} k_b={r['k_b_opt']} period={r['period_at_opt']:.0f}s "
          f"idle={r['idle_s']:.1f}s speedup={r['speedup_vs_npu_only']:.2f}x")
    # NB: the paper's Table 5 lists period 683 s while its own stage times are
    # max(676, 675) = 676 s -> ~7 s of unmodelled per-chunk overhead. Assert the
    # cut exactly, the period loosely, and remember that overhead when comparing.
    ok &= r["k_b_opt"] == 51 and 670 <= r["period_at_opt"] <= 690

    # 2. GPU much slower than NPU (the Tensor G1 risk case): small k_b
    r = solve(180, 3.5, 0.35, 22.0)
    print(f"T_G=10xT_N   : k*={r['k_continuous']:.1f} k_b={r['k_b_opt']} period={r['period_at_opt']:.1f}s "
          f"speedup={r['speedup_vs_npu_only']:.3f}x  ({r['notes']})")
    ok &= 1 <= r["k_b_opt"] <= 40

    # 3. GPU faster than NPU -> GPU should take almost everything
    r = solve(180, 0.2, 1.0, 22.0)
    print(f"T_G<T_N      : k*={r['k_continuous']:.1f} k_b={r['k_b_opt']} speedup={r['speedup_vs_npu_only']:.3f}x")
    ok &= r["k_b_opt"] > 120

    # 4. decoder so big the crossing is off the end
    r = solve(180, 5.0, 0.5, 900.0)
    print(f"huge T_V     : k*={r['k_continuous']:.1f} k_b={r['k_b_opt']} ({r['notes']})")
    ok &= r["k_b_opt"] == 179

    # 5. equal per-block rates -> the cut should split the blocks ~evenly
    r = solve(180, 1.0, 1.0, 20.0)
    print(f"T_G==T_N     : k_b={r['k_b_opt']} speedup={r['speedup_vs_npu_only']:.3f}x")
    ok &= 90 <= r["k_b_opt"] <= 110

    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", help="JSON with {T_G, T_N, T_V, B?} (s or ms; use --unit)")
    ap.add_argument("--Tg", type=float); ap.add_argument("--Tn", type=float)
    ap.add_argument("--Tv", type=float); ap.add_argument("--B", type=int, default=180)
    ap.add_argument("--unit", choices=["s", "ms"], default="s")
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if a.rates:
        d = json.loads(Path(a.rates).read_text(encoding="utf-8"))
        TG, TN, TV, B = d["T_G"], d["T_N"], d["T_V"], d.get("B", 180)
    else:
        TG, TN, TV, B = a.Tg, a.Tn, a.Tv, a.B
        if None in (TG, TN, TV):
            ap.error("give --rates or all of --Tg/--Tn/--Tv")
    if a.unit == "ms":
        TG, TN, TV = TG / 1000, TN / 1000, TV / 1000

    r = solve(B, TG, TN, TV)
    print(f"B={B}  T_G={TG:.3f}s  T_N={TN:.3f}s  T_V={TV:.1f}s")
    print(f"k* (continuous)      : {r['k_continuous']:.2f}   [{r['notes']}]")
    print(f"k_b* (integer)       : {r['k_b_opt']}")
    print(f"period               : {r['period_at_opt']:.2f}s  "
          f"(GPU stage {r['gpu_stage_s']:.2f}s, NPU stage {r['npu_stage_s']:.2f}s, idle {r['idle_s']:.2f}s)")
    print(f"speedup vs NPU-only  : {r['speedup_vs_npu_only']:.3f}x")
    print(f"step-level best      : k_b={r['k_b_step_level']} period={r['period_step_level']:.1f}s "
          f"({r['step_level_regret_pct']:.1f}% worse)")
    print(f"block rates w/o T_V  : k_b={r['k_b_ignoring_TV']} ({r['ignore_TV_regret_pct']:.1f}% worse)")
    if a.json:
        out = dict(r); out.pop("sweep_top10")
        Path(a.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
