#!/usr/bin/env python3
"""Phase 1 deliverable 1: per-output-channel symmetric INT8 base weights -> block regions.

The locked quantisation (configs/design_locks.json) is one FP32 scale per output
channel, `s_j = max_i |W_ij| / 127` with `W` row-major `[N_out, K]` so `K` is
contiguous. This script writes exactly that, and nothing else:

  artifacts/blocks/block_LL.bin         INT8 base GEMM weights, concatenated in
                                        manifest order, each tensor 4096-B aligned
  artifacts/blocks/block_LL.scales.f32  one FP32 scale per output channel, same order
  artifacts/blocks/block_LL.fp16.bin    the block's FP16 remainder (bias, norm weights,
                                        AdaLN scale_shift_table, ffn biases)
  artifacts/top.fp16.bin                the same for the non-block tensors
  artifacts/manifest.json               exact byte offsets of every tensor

Rules this respects:
  * ONE stored INT8 copy of the base weights (1,391,984,640 B = 1.296 GiB). The
    rank-128 expert factors are *not* here: they are FP16 and belong to
    phase1_quant/expert_factors.py, never merged into the INT8 base.
  * LoRA / expert / timestep tensors are skipped by construction (role filter),
    so a second copy of anything cannot be created by this script.
  * Per-GEMM parity (rel. Frobenius error, cosine) is computed in the same pass
    from the dequantised bytes that are actually written, and lands in the
    manifest, so Gate 1 has numbers without a second 5.7 GiB read.

    python phase1_quant/quantize_int8.py                 # all 30 blocks
    python phase1_quant/quantize_int8.py --blocks 0 --out artifacts_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from inspect_weights import role_of
except ImportError:  # imported as a package member
    from phase1_quant.inspect_weights import role_of

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO / "weights_raw" / "DCM_WAN" / "transformer" / "diffusion_pytorch_model.safetensors"
ALIGN = 4096
CLIP = 127
# Gate 1 references (docs/05_phase_plans.md).
GATE1_INT8_BYTES = 1_391_984_640        # base attention + FFN params == INT8 bytes
GATE1_REGION_MIB = 47.29
AHB_CEILING_MIB = 1792.0                # Gate 0: per-block AHB splitting optional, ceiling >> region

# The block's base GEMMs, in the order they are packed into a region. Every one
# of the 30 blocks is shape-identical; this order is the manifest's documented
# concatenation order and the device must use it.
BLOCK_GEMMS = [
    "attn1.to_q.base_layer.weight",   # [1536,1536]
    "attn1.to_k.base_layer.weight",
    "attn1.to_v.base_layer.weight",
    "attn1.to_out.0.base_layer.weight",
    "attn2.to_q.base_layer.weight",
    "attn2.to_k.base_layer.weight",
    "attn2.to_v.base_layer.weight",
    "attn2.to_out.0.base_layer.weight",  # [1536,1536]
    "ffn.net.0.proj.weight",             # [8960,1536]  up
    "ffn.net.2.weight",                  # [1536,8960]  down
]

def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return {k: v for k, v in hdr.items() if k != "__metadata__"}, 8 + n


class Store:
    """Read-only memmap of the safetensors payload; tensors are views, not copies."""

    def __init__(self, path: Path):
        self.hdr, self.data_start = read_header(path)
        self.path = path
        self.arr = np.memmap(path, dtype=np.uint8, mode="r")

    def get(self, name: str) -> np.ndarray:
        m = self.hdr[name]
        if m["dtype"] != "F32":
            raise SystemExit(f"{name}: dtype {m['dtype']} not F32 — quantiser assumes FP32 source")
        a, b = m["data_offsets"]
        flat = np.frombuffer(self.arr, dtype="<f4", count=(b - a) // 4, offset=self.data_start + a)
        return flat.reshape(m["shape"])


def is_base_gemm(name: str, shape) -> bool:
    if len(shape) != 2:
        return False
    if "lora" in name:                       # expert factors: never in the INT8 base
        return False
    return name.endswith(".base_layer.weight") or ".ffn.net.0.proj.weight" in name \
        or ".ffn.net.2.weight" in name


# Role-based classification (configs/block_manifest.json roles, via inspect_weights).
# A suffix-based filter silently dropped the 3-D scale_shift_table, the 5-D patch
# embedding, proj_out.weight and the condition_embedder linears — i.e. tensors the
# locked design says are FP16-kept. Classification must therefore be by ROLE, and
# the run asserts that every tensor in the checkpoint is accounted for exactly once.
EXPERT_ROLES = ("expert_lora_A", "expert_lora_B", "expert_timestep")


def classify(name: str, shape) -> str:
    role = role_of(name)
    if role in EXPERT_ROLES:
        return "expert"                      # deliverable 2: phase1_quant/expert_factors.py
    if role in ("base_attn", "base_ffn") and len(shape) == 2:
        return "int8"
    return "fp16"


def quant_per_output_channel(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """W [N_out, K] -> int8 q [N_out, K] with one FP32 scale per output channel."""
    s = np.abs(W).max(axis=1).astype(np.float32) / CLIP
    s[s == 0.0] = 1.0
    q = np.clip(np.rint(W / s[:, None]), -CLIP, CLIP).astype(np.int8)
    Wd = q.astype(np.float32) * s[:, None]
    nref = float(np.linalg.norm(W.astype(np.float64)))
    d = W.astype(np.float64) - Wd.astype(np.float64)
    stat = {
        "rel_fro": float(np.linalg.norm(d) / nref) if nref else 0.0,
        "cosine": float((W.astype(np.float64) * Wd).sum() / np.linalg.norm(W.astype(np.float64))
                        / np.linalg.norm(Wd.astype(np.float64))) if nref else 1.0,
        "max_abs_err": float(np.abs(d).max()),
        "scale_min": float(s.min()),
        "scale_max": float(s.max()),
        "scale_spread": float(s.max() / s.min()),
        "saturated_frac": float((np.abs(q) == CLIP).mean()),
    }
    return q, s, stat


def align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default=str(REPO / "artifacts"))
    ap.add_argument("--blocks", default="all", help="'all' or a comma list (smoke runs)")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out = Path(args.out)
    (out / "blocks").mkdir(parents=True, exist_ok=True)
    store = Store(ckpt)

    present = {b for b in range(30) if f"blocks.{b}.norm2.weight" in store.hdr}
    if args.blocks == "all":
        blocks = sorted(present)
    else:
        blocks = [int(b) for b in args.blocks.split(",")]
    t0 = time.time()

    manifest = {
        "kind": "phase1_int8_artifact",
        "checkpoint": str(ckpt),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "alignment_bytes": ALIGN,
        "layout": "row-major [N_out, K], K contiguous, int8",
        "quantisation": {
            "scheme": "per-output-channel symmetric int8, s_j = max_i |W_ij| / 127",
            "clip": [-CLIP, CLIP],
            "scales_dtype": "float32 little-endian, one per output channel",
            "fp16_kept": "bias, norm weights, scale_shift_table (AdaLN), ffn biases, io embeddings",
            "expert_factors": "absent by design — FP16, phase1_quant/expert_factors.py",
        },
        "gemm_order": BLOCK_GEMMS,
        "blocks": [],
        "totals": {},
    }

    int8_total = 0
    fp16_total = 0
    fp16_file_total = 0
    n_gemm = 0
    for b in blocks:
        region = bytearray()
        scales = bytearray()
        gemms = []
        for suffix in BLOCK_GEMMS:
            name = f"blocks.{b}.{suffix}"
            if name not in store.hdr:
                raise SystemExit(f"missing expected GEMM {name}")
            W = store.get(name)
            q, s, stat = quant_per_output_channel(W)
            off = align(len(region))
            region.extend(b"\0" * (off - len(region)))
            region.extend(np.ascontiguousarray(q).tobytes(order="C"))
            scale_off = off // 4 * 0 + len(scales)      # byte offset into the .f32 stream
            scales.extend(s.astype("<f4").tobytes())
            gemms.append({
                "name": suffix, "full_name": name, "role": "base_attn" if "attn" in suffix else "base_ffn",
                "shape": list(W.shape), "offset": off, "nbytes": int(q.size),
                "scale_offset": scale_off, "n_scales": int(s.size),
                **stat,
            })
            int8_total += int(q.size)
            n_gemm += 1
            del W, q, s

        region_path = out / "blocks" / f"block_{b:02d}.bin"
        region_path.write_bytes(bytes(region))
        scales_path = out / "blocks" / f"block_{b:02d}.scales.f32"
        scales_path.write_bytes(bytes(scales))

        # FP16 remainder of the block: bias / norm weights / AdaLN tables / ffn biases.
        # classify() puts every non-INT8, non-expert tensor here, whatever its rank.
        fp16 = bytearray()
        fp16_tensors = []
        for name in sorted(store.hdr):
            if not name.startswith(f"blocks.{b}."):
                continue
            meta = store.hdr[name]
            if classify(name, meta["shape"]) != "fp16":
                continue
            W = store.get(name)
            off = align(len(fp16))
            fp16.extend(b"\0" * (off - len(fp16)))
            fp16.extend(W.astype("<f2").tobytes(order="C"))
            fp16_tensors.append({"name": name, "shape": list(W.shape), "dtype": "F16",
                                 "offset": off, "nbytes": int(W.size) * 2})
            fp16_total += int(W.size) * 2
            del W
        fp16_path = out / "blocks" / f"block_{b:02d}.fp16.bin"
        fp16_path.write_bytes(bytes(fp16))
        fp16_file_total += len(fp16)

        manifest["blocks"].append({
            "block": b,
            "region": f"blocks/block_{b:02d}.bin",
            "region_bytes": len(region),
            "region_mib": round(len(region) / 2**20, 3),
            "region_sha256": sha256_file(region_path),
            "scales": f"blocks/block_{b:02d}.scales.f32",
            "scales_bytes": len(scales),
            "scales_sha256": sha256_file(scales_path),
            "fp16": f"blocks/block_{b:02d}.fp16.bin",
            "fp16_bytes": len(fp16),
            "fp16_sha256": sha256_file(fp16_path),
            "gemms": gemms,
            "fp16_tensors": fp16_tensors,
        })
        print(f"block {b:02d}  region {len(region):>9,} B  scales {len(scales):>7,} B  "
              f"fp16 {len(fp16):>7,} B  worst relF "
              f"{max(g['rel_fro'] for g in gemms)*100:5.2f}%  min cos "
              f"{min(g['cosine'] for g in gemms):.6f}  ({time.time()-t0:6.1f}s)")

    # Non-block tensors kept in FP16 (patch/text embedding, proj_out, condition_embedder,
    # norm_out, the top-level AdaLN table). Expert-side LoRA blocks are excluded here —
    # they are deliverable 2.
    top = bytearray()
    top_tensors = []
    for name in sorted(store.hdr):
        if name.startswith("blocks."):
            continue
        meta = store.hdr[name]
        if classify(name, meta["shape"]) != "fp16":
            continue
        W = store.get(name)
        off = align(len(top))
        top.extend(b"\0" * (off - len(top)))
        top.extend(W.astype("<f2").tobytes(order="C"))
        top_tensors.append({"name": name, "shape": list(W.shape), "dtype": "F16",
                            "offset": off, "nbytes": int(W.size) * 2})
        fp16_total += int(W.size) * 2
        del W
    if top:
        (out / "top.fp16.bin").write_bytes(bytes(top))
        fp16_file_total += len(top)
        manifest["top_level_fp16"] = {
            "path": "top.fp16.bin", "bytes": len(top),
            "sha256": sha256_file(out / "top.fp16.bin"), "tensors": top_tensors,
        }

    # Coverage: every tensor in the checkpoint must land in exactly one bucket.
    # This is the check that would have caught the suffix-filter bug: it reported
    # "PASS" on 300 GEMMs while quietly omitting the AdaLN tables and io weights.
    cov = {"int8": [], "fp16": [], "expert": []}
    for name in sorted(store.hdr):
        cov[classify(name, store.hdr[name]["shape"])].append(name)
    written = set()
    for blk in manifest["blocks"]:
        written.update(g["full_name"] for g in blk["gemms"])
        written.update(t["name"] for t in blk["fp16_tensors"])
    written.update(t["name"] for t in top_tensors)
    expected = set(cov["int8"]) | set(cov["fp16"])
    unaccounted = sorted(expected - written)
    phantom = sorted(written - expected)
    if unaccounted or phantom:
        raise SystemExit(f"coverage FAIL: {len(unaccounted)} tensors written nowhere "
                         f"{unaccounted[:5]}, {len(phantom)} written but unknown {phantom[:5]}")
    cov_params = {k: int(sum(int(np.prod(store.hdr[n]['shape'])) for n in v)) for k, v in cov.items()}
    covered = len(written) + len(cov["expert"])
    if covered != len(store.hdr):
        raise SystemExit(f"coverage FAIL: {covered} of {len(store.hdr)} checkpoint tensors accounted for")
    manifest["coverage"] = {
        "checkpoint_tensors": len(store.hdr),
        "int8_tensors": len(cov["int8"]), "int8_params": cov_params["int8"],
        "fp16_tensors": len(cov["fp16"]), "fp16_params": cov_params["fp16"],
        "expert_deferred_tensors": len(cov["expert"]), "expert_deferred_params": cov_params["expert"],
        "unaccounted": 0,
        "note": "expert_* tensors are deliverable 2 (phase1_quant/expert_factors.py), FP16, not in the INT8 base",
    }

    worst = max((g["rel_fro"] for blk in manifest["blocks"] for g in blk["gemms"]), default=0.0)
    mincos = min((g["cosine"] for blk in manifest["blocks"] for g in blk["gemms"]), default=1.0)
    max_region = max((blk["region_mib"] for blk in manifest["blocks"]), default=0.0)
    manifest["totals"] = {
        "blocks": len(manifest["blocks"]),
        "gemms": n_gemm,
        "int8_bytes": int8_total,
        "int8_gib": round(int8_total / 2**30, 4),
        "gate1_int8_bytes": GATE1_INT8_BYTES,
        "gate1_int8_delta_bytes": int8_total - GATE1_INT8_BYTES,
        "fp16_bytes": fp16_total,
        "fp16_file_bytes": fp16_file_total,
        "max_region_mib": max_region,
        "gate1_region_limit_mib": GATE1_REGION_MIB,
        "ahb_ceiling_mib": AHB_CEILING_MIB,
        "worst_gemm_rel_fro": worst,
        "min_gemm_cosine": mincos,
        "cosine_bar_0p999_met": bool(mincos >= 0.999),
        "int8_bytes_match": bool(abs(int8_total - GATE1_INT8_BYTES) <= 2**20),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    t = manifest["totals"]
    print()
    print(f"INT8 base total   : {t['int8_bytes']:,} B = {t['int8_gib']:.4f} GiB  "
          f"(Gate 1 {GATE1_INT8_BYTES:,} B, delta {t['gate1_int8_delta_bytes']:+,} B, "
          f"{'MATCH' if t['int8_bytes_match'] else 'MISMATCH'})")
    print(f"FP16 remainder    : {t['fp16_bytes']:,} B = {t['fp16_bytes']/2**20:.2f} MiB "
          f"({t['fp16_file_bytes']:,} B on disk with alignment; blocks + top-level, expert factors excluded)")
    c = manifest["coverage"]
    print(f"coverage          : {c['checkpoint_tensors']} checkpoint tensors = "
          f"{c['int8_tensors']} INT8 ({c['int8_params']:,} params) + {c['fp16_tensors']} FP16 "
          f"({c['fp16_params']:,}) + {c['expert_deferred_tensors']} expert-side deferred "
          f"({c['expert_deferred_params']:,}), 0 unaccounted")
    print(f"max block region  : {t['max_region_mib']:.2f} MiB "
          f"(Gate 1 {GATE1_REGION_MIB} MiB, AHB {AHB_CEILING_MIB:.0f} MiB)")
    print(f"worst GEMM relF   : {t['worst_gemm_rel_fro']*100:.2f}%   "
          f"min cosine {t['min_gemm_cosine']:.6f}  "
          f"({'PASS' if t['cosine_bar_0p999_met'] else 'FAIL'} vs 0.999 bar)")
    print(f"({t['elapsed_s']}s, {n_gemm} GEMMs, host only — no device number claimed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
