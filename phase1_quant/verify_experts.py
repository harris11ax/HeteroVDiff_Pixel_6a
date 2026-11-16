#!/usr/bin/env python3
"""Independent verifier for Phase 1 deliverable 2 (artifacts/experts/detail/).

`expert_factors.py`'s own report is not the evidence. This pass re-reads the **bytes on disk**,
re-derives the FP16 payload from the FP32 checkpoint, and demands byte identity — the same
standard `phase1_quant/verify_artifact.py` holds the INT8 base to. It shares no writer state: it
re-opens the safetensors, re-selects the expert tensors by role, and rebuilds every tensor it
compares.

    python phase1_quant/verify_experts.py

Checks, in order (any failure exits non-zero):
  1. every file listed in the manifest exists and its sha256 matches
  2. every tensor's on-disk bytes at (offset, nbytes) equal a fresh FP32->FP16 cast of the source
  3. shapes / rank / K / N_out in the manifest match the checkpoint header
  4. the selection is complete: 492 expert-side tensors, every one present exactly once
  5. totals: 240,089,216 B payload, 30 block files + 1 top-level file
  6. one-copy rule: no tensor verified here also appears in the INT8 base manifest
  7. routing truth is present and consistent with the locked schedule
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "weights_raw" / "DCM_WAN" / "transformer" / "diffusion_pytorch_model.safetensors"
EXP_DIR = REPO / "artifacts" / "experts" / "detail"
BASE_MANIFEST = REPO / "artifacts" / "manifest.json"
EXPECTED_BYTES = 240_089_216
EXPECTED_TENSORS = 492


def role_of(name: str) -> str:
    """Deliberately re-implemented (not imported) so a classifier bug cannot verify itself."""
    n = name.lower()
    if "lora_a" in n:
        return "expert_lora_A"
    if "lora_b" in n:
        return "expert_lora_B"
    if any(t in n for t in ("condition_embedder_lora", "norm_out_lora", "proj_out_lora")):
        return "expert_timestep"
    return "base_or_fp16"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    fails: list[str] = []

    with open(CKPT, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    data_start = 8 + n
    hdr = {k: v for k, v in hdr.items() if k != "__metadata__"}
    src = np.memmap(CKPT, dtype=np.uint8, mode="r")

    man = json.loads((EXP_DIR / "manifest.json").read_text(encoding="utf-8"))
    routing = json.loads((EXP_DIR / "routing.json").read_text(encoding="utf-8"))

    def source_fp16_bytes(name: str) -> bytes:
        m = hdr[name]
        a, b = m["data_offsets"]
        flat = np.frombuffer(src, dtype="<f4", count=(b - a) // 4, offset=data_start + a)
        W = flat.reshape(m["shape"])
        return np.ascontiguousarray(W, dtype="<f2").tobytes(order="C")

    # ---- 1. files + sha256 ------------------------------------------------
    files = [(man["blocks"][i]["path"], man["blocks"][i]["bytes"], man["blocks"][i]["sha256"])
             for i in range(len(man["blocks"]))]
    if "top_level" in man:
        files.append((man["top_level"]["path"], man["top_level"]["bytes"], man["top_level"]["sha256"]))
    disk_total = 0
    for rel, nbytes, sha in files:
        p = REPO / "artifacts" / rel
        if not p.exists():
            fails.append(f"missing file {rel}")
            continue
        sz = p.stat().st_size
        disk_total += sz
        if sz != nbytes:
            fails.append(f"{rel}: size {sz:,} != manifest {nbytes:,}")
        got = sha256_file(p)
        if got != sha:
            fails.append(f"{rel}: sha256 {got[:16]} != manifest {sha[:16]}")

    # ---- 2-3. byte identity + shape truth --------------------------------
    checked = 0
    payload = 0
    names: list[str] = []
    for group in [b for b in man["blocks"]] + ([man["top_level"]] if "top_level" in man else []):
        blob = (REPO / "artifacts" / group["path"]).read_bytes()
        for rec in group["tensors"]:
            name, off, nb = rec["name"], rec["offset"], rec["nbytes"]
            names.append(name)
            if name not in hdr:
                fails.append(f"{name}: not in checkpoint header")
                continue
            expect = source_fp16_bytes(name)
            if nb != len(expect):
                fails.append(f"{name}: nbytes {nb} != 2 x params {len(expect)}")
            if list(hdr[name]["shape"]) != rec["shape"]:
                fails.append(f"{name}: manifest shape {rec['shape']} != header {hdr[name]['shape']}")
            if blob[off:off + nb] != expect:
                fails.append(f"{name}: payload bytes differ from a fresh FP32->FP16 cast")
            if rec["role"] == "expert_lora_A" and (rec.get("rank") != 128 or rec["shape"][0] != 128):
                fails.append(f"{name}: rank/shape not [128,1536]")
            if rec["role"] == "expert_lora_B" and (rec.get("rank") != 128 or rec["shape"][1] != 128):
                fails.append(f"{name}: rank/shape not [1536,128]")
            payload += nb
            checked += 1

    # ---- 4. selection completeness ---------------------------------------
    expected = {k for k in hdr if role_of(k) != "base_or_fp16"}
    if len(expected) != EXPECTED_TENSORS:
        fails.append(f"checkpoint expert-side tensor count {len(expected)} != {EXPECTED_TENSORS}")
    missing = sorted(expected - set(names))
    extra = sorted(set(names) - expected)
    if missing:
        fails.append(f"{len(missing)} expert tensors never written: {missing[:5]}")
    if extra:
        fails.append(f"{len(extra)} written tensors are not expert-side: {extra[:5]}")
    if len(names) != len(set(names)):
        fails.append("duplicate tensor names across the artifact")

    # ---- 5. totals --------------------------------------------------------
    if payload != EXPECTED_BYTES:
        fails.append(f"payload {payload:,} B != {EXPECTED_BYTES:,} B")
    if man["totals"]["fp16_bytes"] != payload:
        fails.append(f"manifest totals {man['totals']['fp16_bytes']:,} != re-derived {payload:,}")
    if len(man["blocks"]) != 30:
        fails.append(f"{len(man['blocks'])} block files != 30")

    # ---- 6. one-copy rule ------------------------------------------------
    if BASE_MANIFEST.exists():
        base = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
        base_names = {g["full_name"] for blk in base["blocks"] for g in blk["gemms"]}
        base_names |= {t["name"] for blk in base["blocks"] for t in blk["fp16_tensors"]}
        base_names |= {t["name"] for t in base.get("top_level_fp16", {}).get("tensors", [])}
        both = sorted(set(names) & base_names)
        if both:
            fails.append(f"one-copy rule: {len(both)} tensors in both artifacts {both[:5]}")
    else:
        fails.append(f"INT8 base manifest absent at {BASE_MANIFEST} — cannot check the one-copy rule")

    # ---- 7. routing -------------------------------------------------------
    steps = routing.get("steps", [])
    if [s["expert"] for s in steps] != ["semantic"] * 3 + ["detail"] * 3:
        fails.append(f"routing schedule {[s['expert'] for s in steps]} != semantic x3 + detail x3")
    if not routing.get("threshold_agrees"):
        fails.append("routing: timestep threshold and kappa switch disagree")
    if steps and steps[-1]["block_exec_range"][1] != 180:
        fails.append(f"routing covers {steps[-1]['block_exec_range'][1]} block executions != 180")

    # ---- verdict ---------------------------------------------------------
    print(f"files verified    : {len(files)} ({disk_total:,} B on disk, {payload:,} B payload)")
    print(f"tensors verified  : {checked} ({len(expected)} expert-side in checkpoint, "
          f"{len(missing)} unwritten, {len(extra)} foreign)")
    print(f"byte identity     : {'PASS' if not fails else 'FAIL'} — payload re-derived from FP32 source")
    print(f"payload           : {payload:,} B vs Gate 1 {EXPECTED_BYTES:,} B "
          f"(delta {payload - EXPECTED_BYTES:+,})")
    print(f"routing           : {len(steps)} steps, {steps[-1]['block_exec_range'][1] if steps else 0} block executions")
    if fails:
        print(f"\nVERDICT: FAIL — {len(fails)} problem(s)")
        for f_ in fails[:20]:
            print(f"  - {f_}")
        return 1
    print("\nVERDICT: PASS — the expert artifact is byte-identical to a fresh FP16 cast of the "
          "checkpoint, complete over all 492 expert-side tensors, and disjoint from the INT8 base")
    return 0


if __name__ == "__main__":
    sys.exit(main())
