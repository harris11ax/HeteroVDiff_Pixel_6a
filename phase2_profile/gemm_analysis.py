#!/usr/bin/env python3
"""Turn a gemm_mali_<ts>.json run into the per-DiT-block table.

Every number is read from the JSON that `tools/gemm/gemm_host.py` wrote from the
phone; nothing here models a rate. The only arithmetic is
    block_MACs = M * (8*1536*1536 + 2*8960*1536)
and, where a per-GEMM time is used, summing 8 x attn + ffn_up + ffn_down.

    python phase2_profile/gemm_analysis.py docs/device/gemm_mali_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

# Preset A: F_z=3, H_z=34, W_z=60, patch (1,2,2) -> 1530 tokens
M_PRESET_A = 1530
GEMMS_PER_BLOCK = 8 * 1536 * 1536 + 2 * 8960 * 1536      # base MACs, per token

# bounds already computed in docs/device/RESULTS_gate0.md (arithmetic, not measured)
BOUND_TPU_MS = 532.0        # 151.7 GFLOP at 285 GOPS
BOUND_NAIVE_MS = 14600.0    # 151.7 GFLOP at 10.4 GOPS
BOUND_1TOPS_MS = 152.0      # 151.7 GFLOP at 1 TOPS (the hypothetical)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def block_rows(rec: dict) -> list[dict]:
    out = []
    for b in rec["parsed"]["block"]:
        out.append({"variant": b["variant"], "lora": b["lora"], "ms": b["ms"],
                    "gops": b["gops"], "gmac": b["gmac"]})
    return out


def sweep_best(rec: dict, shape: str, M: int) -> dict | None:
    cands = [g for g in rec["parsed"]["gemm"] if g["shape"] == shape and g["M"] == M]
    if not cands:
        return None
    return max(cands, key=lambda g: g["gops"])


def composite_block_ms(rec: dict, M: int) -> tuple[float | None, str | None]:
    """8*attn + ffn_up + ffn_down from the per-GEMM sweep, for the best variant."""
    picks = {}
    for sh in ("attn", "ffn_up", "ffn_down"):
        b = sweep_best(rec, sh, M)
        if not b:
            return None, None
        picks[sh] = b
    # all three must come from the same variant to be a fair composite
    names = {b["variant"] for b in picks.values()}
    if len(names) != 1:
        name = max(names, key=lambda n: sum(b["gops"] for b in picks.values() if b["variant"] == n))
        for sh in picks:
            cands = [g for g in rec["parsed"]["gemm"]
                     if g["shape"] == sh and g["M"] == M and g["variant"] == name]
            if not cands:
                return None, None
            picks[sh] = max(cands, key=lambda g: g["gops"])
    ms = 8 * picks["attn"]["ms"] + picks["ffn_up"]["ms"] + picks["ffn_down"]["ms"]
    return ms, picks["attn"]["variant"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="gemm_mali_<ts>.json files")
    ap.add_argument("--M", type=int, default=M_PRESET_A)
    a = ap.parse_args()

    recs = [(Path(p).name, load(Path(p))) for p in a.runs]

    print(f"# Mali-G78 INT8 GEMM — per-block analysis (M = {a.M})\n")
    macs = a.M * GEMMS_PER_BLOCK
    print(f"one DiT block at Preset A = {macs/1e9:.3f} GMAC = {2*macs/1e9:.1f} GFLOP "
          f"(base weights; LoRA delta counted separately)\n")

    for name, rec in recs:
        d = rec["parsed"]["device"]
        print(f"## {name}  (kernel {rec['kernel_sha256']}, iters={rec['iters']})")
        tb, ta = rec["temps"]["before"], rec["temps"]["after"]
        keep = ("TPU", "skin_therm1", "skin_therm2", "battery", "VIRTUAL-SKIN")
        print("thermal before: " + " ".join(f"{k}={tb[k]:.1f}" for k in keep if k in tb))
        print("thermal after : " + " ".join(f"{k}={ta[k]:.1f}" for k in keep if k in ta))
        print()

        rows = block_rows(rec)
        if rows:
            print("| variant | lora mode | ms/block | GOPS | x TPU bound | x naive bound |")
            print("|---|---|---|---|---|---|")
            for r in rows:
                print(f"| `{r['variant']}` | {r['lora']} | **{r['ms']:.1f}** | {r['gops']:.0f} | "
                      f"{BOUND_TPU_MS/r['ms']:.2f}x | {BOUND_NAIVE_MS/r['ms']:.0f}x |")
            print()

        ms, var = composite_block_ms(rec, a.M)
        if ms:
            print(f"composite (sum of measured per-GEMM times, `{var}`): **{ms:.1f} ms/block** "
                  f"= {2*macs/(ms*1e6):.0f} GOPS")
            print(f"  vs TPU bound {BOUND_TPU_MS:.0f} ms -> {BOUND_TPU_MS/ms:.2f}x faster")
            print(f"  vs naive Mali {BOUND_NAIVE_MS:.0f} ms -> {BOUND_NAIVE_MS/ms:.0f}x faster")
            print(f"  vs 1 TOPS bound {BOUND_1TOPS_MS:.0f} ms -> {BOUND_1TOPS_MS/ms:.2f}x")
            print()

        f16 = rec["parsed"].get("f16") or []
        if f16:
            tot = sum(float(f["ms"]) for f in f16)
            print("LoRA delta as counted FP16 GEMMs (per attention projection): "
                  + " + ".join(f"{float(f['ms']):.1f} ms" for f in f16)
                  + f" = **{tot:.1f} ms**")
            print(f"  x8 projections = {8*tot:.0f} ms/block for the detail expert "
                  f"(steps 3-5 of the 6-step trajectory)\n")
        s8 = []
        for sh in ("lora_up8", "lora_dn8"):
            b = sweep_best(rec, sh, a.M)
            if b:
                s8.append(b)
        if s8:
            tot8 = sum(b["ms"] for b in s8)
            print("same delta with INT8 factor storage (114.5 MiB instead of 229.0 MiB): "
                  + " + ".join(f"{b['ms']:.1f} ms" for b in s8) + f" = **{tot8:.1f} ms**")
            print(f"  x8 projections = {8*tot8:.0f} ms/block\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
