#!/usr/bin/env python3
"""Phase 1 deliverable 2: the detail expert's factorised weights, FP16, as their own artifact.

The locked design stores the detail expert **factorised** — rank-128 LoRA `A [128,1536]` /
`B [1536,128]` per attention projection, plus the expert's own timestep / text-conditioning /
output-projection blocks — in **FP16, never merged into the INT8 base**. This script writes exactly
that and nothing else:

  artifacts/experts/detail/block_LL.fp16.bin   the block's 8 projections x (A then B), FP16,
                                               4096-B aligned, row-major [N_out, K]
  artifacts/experts/detail/top.fp16.bin        condition_embedder_lora (10 tensors) +
                                               proj_out_lora (2 tensors)
  artifacts/experts/detail/manifest.json       per-tensor offsets, shapes, FP16 parity, sha256
  artifacts/experts/detail/routing.json        steps -> expert (semantic x3, detail x3)

Rules this respects:
  * **One stored INT8 copy of the base weights.** Nothing here is INT8, and no base GEMM tensor is
    read at all: the selection is by role (`inspect_weights.role_of`), the same classifier the
    quantiser uses, and the run asserts that the two artifacts' tensor sets are disjoint.
  * **Coverage is asserted.** 492 checkpoint tensors are expert-side; every one must be written
    exactly once, or the script fails. (Gate 1's suffix-filter bug — 276,480 AdaLN params written
    to no file while the script printed PASS — is the reason this is an assertion, not a print.)
  * **Nothing is invented.** The LoRA scaling constant (`alpha/rank`) is *not* in this checkpoint;
    it is recorded as `null` with the reason, to be taken from the reference repo, not guessed.
  * Per-tensor FP16 parity (rel. Frobenius, cosine) is computed from the bytes actually written.

    python phase1_quant/expert_factors.py                # all 30 blocks + top level
    python phase1_quant/expert_factors.py --blocks 0 --out artifacts_smoke
    python phase1_quant/verify_experts.py                # independent re-derivation
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from inspect_weights import role_of
    from quantize_int8 import Store, align, sha256_file, BLOCK_GEMMS
except ImportError:  # imported as a package member
    from phase1_quant.inspect_weights import role_of
    from phase1_quant.quantize_int8 import Store, align, sha256_file, BLOCK_GEMMS

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO / "weights_raw" / "DCM_WAN" / "transformer" / "diffusion_pytorch_model.safetensors"
LOCKS = REPO / "configs" / "design_locks.json"

# Gate 1 reference figure (docs/device/RESULTS_gate1.md, configs/design_locks.json
# expert_factors_mib_fp16 = 229.0): 120,044,608 expert-side params x 2 B.
EXPERT_FP16_BYTES = 240_089_216
EXPERT_PARAMS = EXPERT_FP16_BYTES // 2      # 120,044,608
RANK = 128
D_MODEL = 1536

EXPERT_ROLES = ("expert_lora_A", "expert_lora_B", "expert_timestep")

# The 8 LoRA-wrapped attention projections of a DiT block, in the same order the INT8 base uses
# (quantize_int8.BLOCK_GEMMS). The FFN has no LoRA in this checkpoint (its weights are plain
# `.weight`, i.e. shared base) — verified, not assumed: 480 block LoRA tensors / 30 = 16 = 8 x (A,B).
BLOCK_PROJECTIONS = [
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
]
LORA_A = "lora_A.lora1.weight"
LORA_B = "lora_B.lora1.weight"


def fp16_stat(W32: np.ndarray, W16: np.ndarray) -> dict:
    """Parity of the stored FP16 payload against the FP32 checkpoint tensor."""
    a = W32.astype(np.float64)
    b = W16.astype(np.float64)
    nref = float(np.linalg.norm(a))
    d = a - b
    nb = float(np.linalg.norm(b))
    return {
        "rel_fro": float(np.linalg.norm(d) / nref) if nref else 0.0,
        "cosine": float((a * b).sum() / nref / nb) if (nref and nb) else 1.0,
        "max_abs_err": float(np.abs(d).max()) if d.size else 0.0,
        "abs_max": float(np.abs(a).max()) if a.size else 0.0,
        "dtype_min_fp16": bool(np.isfinite(W16.astype(np.float32)).all()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default=str(REPO / "artifacts"))
    ap.add_argument("--blocks", default="all", help="'all' or a comma list (smoke runs)")
    args = ap.parse_args()

    ckpt, out = Path(args.ckpt), Path(args.out)
    exp_dir = out / "experts" / "detail"
    exp_dir.mkdir(parents=True, exist_ok=True)
    store = Store(ckpt)
    locks = json.loads(LOCKS.read_text(encoding="utf-8"))

    present = {b for b in range(30) if f"blocks.{b}.norm2.weight" in store.hdr}
    blocks = sorted(present) if args.blocks == "all" else [int(b) for b in args.blocks.split(",")]
    t0 = time.time()

    manifest = {
        "kind": "phase1_expert_factors",
        "expert": "detail",
        "checkpoint": str(ckpt),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "storage": {
            "dtype": "float16 little-endian",
            "alignment_bytes": 4096,
            "layout": "row-major [N_out, K], K contiguous (A: [128,1536], B: [1536,128])",
            "merged_into_int8_base": False,
            "note": "locked design: factorised, FP16, its own artifact — never merged (design_locks.json quantisation.expert_factors)",
        },
        "lora": {
            "rank": RANK,
            "scaling_alpha": None,
            "scaling_note": "no alpha/scale tensor exists in this checkpoint (only .weight); the runtime scaling "
                            "constant must come from the reference repo — NOT guessed here",
            "projections_per_block": BLOCK_PROJECTIONS,
            "ffn_lora": False,
            "ffn_lora_note": "this checkpoint's FFN weights are plain .weight (shared base); the 480 block LoRA "
                             "tensors are 30 blocks x 8 projections x (A,B), verified by count",
        },
        "runtime_cost_measured": locks.get("mali_int8_gemm", {}).get("lora_delta_ms_per_block", {}),
        "blocks": [],
        "totals": {},
    }

    payload_total = 0
    file_total = 0
    n_tensors = 0
    written: set[str] = set()

    for b in blocks:
        buf = bytearray()
        recs = []
        for proj in BLOCK_PROJECTIONS:
            for suffix, role, shape_expect in (
                (LORA_A, "expert_lora_A", [RANK, D_MODEL]),
                (LORA_B, "expert_lora_B", [D_MODEL, RANK]),
            ):
                name = f"blocks.{b}.{proj}.{suffix}"
                if name not in store.hdr:
                    raise SystemExit(f"missing expected LoRA tensor {name}")
                if role_of(name) != role:
                    raise SystemExit(f"{name}: role {role_of(name)} != {role}")
                W = store.get(name)
                if list(W.shape) != shape_expect:
                    raise SystemExit(f"{name}: shape {list(W.shape)} != {shape_expect}")
                W16 = np.ascontiguousarray(W, dtype="<f2")
                off = align(len(buf))
                buf.extend(b"\0" * (off - len(buf)))
                buf.extend(W16.tobytes(order="C"))
                recs.append({
                    "name": name, "projection": proj, "role": role,
                    "shape": list(W.shape), "offset": off, "nbytes": int(W16.size) * 2,
                    "rank": RANK, "k": int(W.shape[1]), "n_out": int(W.shape[0]),
                    **fp16_stat(W, W16),
                })
                payload_total += int(W16.size) * 2
                n_tensors += 1
                written.add(name)
                del W, W16

        path = exp_dir / f"block_{b:02d}.fp16.bin"
        path.write_bytes(bytes(buf))
        file_total += len(buf)
        manifest["blocks"].append({
            "block": b, "path": f"experts/detail/block_{b:02d}.fp16.bin",
            "bytes": len(buf), "sha256": sha256_file(path), "tensors": recs,
        })
        worst = max(r["rel_fro"] for r in recs)
        print(f"block {b:02d}  {len(buf):>9,} B  16 tensors  worst FP16 relF {worst*100:.4f}%  "
              f"({time.time()-t0:5.1f}s)")

    # Top level: the detail expert's own timestep / text-conditioning / output projection.
    top = bytearray()
    top_recs = []
    for name in sorted(store.hdr):
        if name.startswith("blocks."):
            continue
        meta = store.hdr[name]
        if role_of(name) not in EXPERT_ROLES:
            continue
        W = store.get(name)
        W16 = np.ascontiguousarray(W, dtype="<f2")
        off = align(len(top))
        top.extend(b"\0" * (off - len(top)))
        top.extend(W16.tobytes(order="C"))
        top_recs.append({
            "name": name, "role": role_of(name), "shape": list(W.shape),
            "offset": off, "nbytes": int(W16.size) * 2, **fp16_stat(W, W16),
        })
        payload_total += int(W16.size) * 2
        n_tensors += 1
        written.add(name)
        del W, W16
    if top_recs:
        (exp_dir / "top.fp16.bin").write_bytes(bytes(top))
        file_total += len(top)
        manifest["top_level"] = {
            "path": "experts/detail/top.fp16.bin", "bytes": len(top),
            "sha256": sha256_file(exp_dir / "top.fp16.bin"), "tensors": top_recs,
        }

    # ---- assertions -------------------------------------------------------
    # 1. Coverage: every expert-role tensor in the checkpoint is written exactly once.
    expected = {n for n in store.hdr if role_of(n) in EXPERT_ROLES}
    missing = sorted(expected - written)
    phantom = sorted(written - expected)
    if missing or phantom:
        raise SystemExit(f"coverage FAIL: {len(missing)} expert tensors unwritten {missing[:5]}, "
                         f"{len(phantom)} written but not expert-role {phantom[:5]}")

    # 2. Disjointness with the INT8 base: no tensor here may also be an INT8 base GEMM.
    base_names = {f"blocks.{b}.{s}" for b in range(30) for s in BLOCK_GEMMS}
    overlap = sorted(written & base_names)
    if overlap:
        raise SystemExit(f"one-copy rule FAIL: {len(overlap)} tensors are in both artifacts {overlap[:5]}")

    # 3. The Gate 1 byte figure.
    delta = payload_total - EXPERT_FP16_BYTES
    if abs(delta) > 2**20:
        raise SystemExit(f"FAIL: expert FP16 payload {payload_total:,} B, expected {EXPERT_FP16_BYTES:,} B "
                         f"(delta {delta:+,})")

    # 4. Routing table: steps -> expert, from the locked design.
    model = locks["model"]
    steps = []
    per_step = model["L_blocks"]
    for s, (t, expert) in enumerate(zip(model["timesteps"], model["expert_schedule"])):
        steps.append({
            "step": s, "timestep": t, "expert": expert,
            "block_exec_range": [s * per_step, (s + 1) * per_step],
            "timestep_below_detail_threshold": bool(t < model["detail_expert_threshold_t"]),
        })
    routing = {
        "kind": "phase1_expert_routing",
        "source": "configs/design_locks.json model.timesteps / expert_schedule / kappa_switch_step",
        "kappa_switch_step": model["kappa_switch_step"],
        "detail_expert_threshold_t": model["detail_expert_threshold_t"],
        "schedule": model["expert_schedule"],
        "steps": steps,
        "check": "the timestep-based threshold and the kappa switch agree on every step",
        "threshold_agrees": all(s["expert"] == ("detail" if s["timestep"] < model["detail_expert_threshold_t"]
                                                else "semantic") for s in steps),
        "block_executions": steps[-1]["block_exec_range"][1] if steps else 0,
    }
    if not routing["threshold_agrees"]:
        raise SystemExit("routing FAIL: expert_schedule disagrees with detail_expert_threshold_t on some step")
    (exp_dir / "routing.json").write_text(json.dumps(routing, indent=2), encoding="utf-8")

    all_recs = [r for blk in manifest["blocks"] for r in blk["tensors"]] + top_recs
    manifest["totals"] = {
        "blocks": len(manifest["blocks"]),
        "tensors": n_tensors,
        "params": payload_total // 2,
        "fp16_bytes": payload_total,
        "fp16_mib": round(payload_total / 2**20, 2),
        "file_bytes": file_total,
        "alignment_pad_bytes": file_total - payload_total,
        "expected_fp16_bytes": EXPERT_FP16_BYTES,
        "delta_bytes": delta,
        "match": bool(abs(delta) <= 2**20),
        "int8_bytes_written": 0,
        "worst_rel_fro": max(r["rel_fro"] for r in all_recs),
        "min_cosine": min(r["cosine"] for r in all_recs),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    t = manifest["totals"]
    print()
    print(f"expert factors    : {t['fp16_bytes']:,} B = {t['fp16_mib']:.2f} MiB FP16 "
          f"(Gate 1 {EXPERT_FP16_BYTES:,} B, delta {t['delta_bytes']:+,} B, "
          f"{'MATCH' if t['match'] else 'MISMATCH'})")
    print(f"tensors           : {t['tensors']} = {len(manifest['blocks'])} blocks x 16 + "
          f"{len(top_recs)} top-level, every expert-role tensor in the checkpoint, 0 unaccounted")
    print(f"files on disk     : {t['file_bytes']:,} B ({t['alignment_pad_bytes']:,} B alignment pad)")
    print(f"FP16 parity       : worst relF {t['worst_rel_fro']*100:.4f}%  min cosine {t['min_cosine']:.7f}")
    print(f"routing           : {routing['block_executions']} block executions, "
          f"kappa_switch_step {routing['kappa_switch_step']}, threshold agrees = {routing['threshold_agrees']}")
    print(f"int8 bytes written: 0 (one stored INT8 copy rule intact)   ({t['elapsed_s']}s, host only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
