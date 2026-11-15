#!/usr/bin/env python3
"""Inventory the DCM-Wan2.1-1.3B checkpoint without torch.

Reads only the safetensors header (no tensors are materialised), so it runs in
a second on any machine. Emits the facts Phase 1 depends on:

  * total parameter count and per-block footprint (one DiT block == one
    AHardwareBuffer region, see blueprint §1 row 2 / paper §4.2)
  * which tensors are shared base vs expert-specific (detail expert = rank-128
    LoRA + its own timestep embedding, paper §4.2 eq. 3)
  * exact GEMM shapes for the OpenCL / NNAPI kernel tiling
  * INT8 / FP16 / FP32 size projections and per-region size
  * a machine-readable manifest (--manifest) that phase1_quant/quantize_int8.py
    consumes: every tensor, its role, shape, dtype and byte size

    python phase1_quant/inspect_weights.py --manifest configs/block_manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO / "weights_raw" / "DCM_WAN" / "transformer" / "diffusion_pytorch_model.safetensors"
DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1, "F64": 8, "I64": 8}


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def numel(shape) -> int:
    v = 1
    for s in shape:
        v *= s
    return v


def role_of(name: str) -> str:
    """Paper §4.2 split: shared base (INT8, one copy) vs expert factors (kept aside)."""
    n = name.lower()
    if "lora_a" in n:
        return "expert_lora_A"
    if "lora_b" in n:
        return "expert_lora_B"
    if "condition_embedder_lora" in n or "norm_out_lora" in n or "proj_out_lora" in n:
        return "expert_timestep"          # detail expert's own timestep/proj block
    if "ffn" in n and n.endswith("weight"):
        return "base_ffn"                 # Wan FFN has no LoRA wrapper: plain .weight
    if "base_layer.weight" in n:
        if "attn" in n:
            return "base_attn"
        return "base_other"
    if "base_layer.bias" in n:
        return "bias"
    if "norm_q" in n or "norm_k" in n or name.endswith(".norm2.weight") or name.endswith(".norm2.bias"):
        return "norm"
    if "scale_shift_table" in n:
        return "adaln"
    if n.startswith("patch_embedding") or n.startswith("text_embedding"):
        return "io_embed"
    if n.startswith("condition_embedder") or n.startswith("norm_out") or n.startswith("proj_out"):
        return "io_embed"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--manifest", default=None, help="write per-block tensor manifest JSON here")
    args = ap.parse_args()

    path = Path(args.ckpt)
    hdr = read_header(path)
    tensors = {k: v for k, v in hdr.items() if k != "__metadata__"}

    total = 0
    by_role = defaultdict(int)
    per_block = defaultdict(int)
    block_tensors: dict[int, list] = defaultdict(list)
    top = []
    dtypes = defaultdict(int)

    for name, meta in tensors.items():
        n = numel(meta["shape"])
        role = role_of(name)
        total += n
        by_role[role] += n
        dtypes[meta["dtype"]] += n
        rec = {"name": name, "shape": meta["shape"], "dtype": meta["dtype"],
               "params": n, "bytes": n * DTYPE_BYTES.get(meta["dtype"], 4), "role": role}
        m = re.match(r"blocks\.(\d+)\.", name)
        if m:
            b = int(m.group(1))
            per_block[b] += n
            block_tensors[b].append(rec)
        else:
            top.append(rec)

    print(f"checkpoint          : {path.name}  ({path.stat().st_size/2**30:.2f} GiB on disk)")
    print(f"tensors             : {len(tensors)}   dtypes: {dict(dtypes)}")
    print(f"total parameters    : {total:,}  ({total/1e9:.4f} B)")
    print()
    print("roles (params):")
    for r, n in sorted(by_role.items(), key=lambda kv: -kv[1]):
        print(f"   {r:18s} {n:>13,}  ({100*n/total:5.2f}%)  FP16 {n*2/2**30:6.3f} GiB  INT8 {n/2**30:6.3f} GiB")
    print()
    print(f"DiT blocks          : {len(per_block)}  (blueprint L = 30)")
    if per_block:
        vals = set(per_block.values())
        f = max(vals)
        print(f"params per block    : {f:,}   uniform={len(vals)==1}")
        print(f"INT8 region / block : {f/2**20:.2f} MiB   x{len(per_block)} = {sum(per_block.values())/2**30:.3f} GiB")
    print(f"base INT8 artifact  : {(by_role['base_attn']+by_role['base_ffn']+by_role['base_other'])/2**30:.3f} GiB "
          f"(blueprint: ~1.47 GiB)")
    print(f"expert factors      : {(by_role['expert_lora_A']+by_role['expert_lora_B']+by_role['expert_timestep'])/2**20:.1f} MiB "
          f"(paper: 94.4M LoRA + 25.6M timestep)")
    print()
    print("block-0 GEMMs (kernel tiling targets):")
    for r in sorted(block_tensors.get(0, []), key=lambda r: -r["params"]):
        if len(r["shape"]) == 2 and r["params"] > 1_000_000:
            print(f"   {r['name']:52s} {str(r['shape']):14s} {r['params']/2**20:7.2f} M")
    print()
    print("top-level tensors (non-block):")
    for r in sorted(top, key=lambda r: -r["params"])[:12]:
        print(f"   {r['name']:52s} {str(r['shape']):18s} {r['params']/2**20:8.3f} M  [{r['role']}]")
    if len(top) > 12:
        print(f"   ... {len(top)-12} more")

    if args.manifest:
        out = {
            "checkpoint": str(path), "total_params": total,
            "dtype_params": dict(dtypes),
            "role_params": dict(by_role),
            "blocks": {str(b): block_tensors[b] for b in sorted(block_tensors)},
            "top_level": top,
        }
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
