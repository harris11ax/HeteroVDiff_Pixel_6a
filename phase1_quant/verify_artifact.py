#!/usr/bin/env python3
"""Gate 1 (artifact integrity) — an INDEPENDENT pass over what phase1_quant/quantize_int8.py wrote.

The quantiser's own manifest is a self-report. This script re-derives ground truth:

  1. region / scales / fp16 file sizes, 4096-B alignment and SHA-256 against the manifest;
  2. the INT8 bytes on disk re-quantised from the FP32 checkpoint and compared
     **byte for byte** (not "close" — identical, that is what integrity means);
  3. dequant(int8 on disk) vs W: per-GEMM relative Frobenius error, cosine, saturation;
  4. the FP16 remainder's own error (it is not exact either: bias/norm/AdaLN are
     cast to FP16, so the report states what that costs);
  5. the Gate 1 bars: total INT8 bytes, per-region size, cosine >= 0.999.

Reads only files + the checkpoint; no device, no network.

    python phase1_quant/verify_artifact.py                 # all blocks
    python phase1_quant/verify_artifact.py --blocks 0,29   # spot check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "phase1_quant"))
from quantize_int8 import (  # noqa: E402  (same quantiser definition, deliberately)
    CLIP,
    GATE1_INT8_BYTES,
    GATE1_REGION_MIB,
    ALIGN,
    Store,
    quant_per_output_channel,
    DEFAULT_CKPT,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=str(REPO / "artifacts"))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    adir = Path(args.artifact)
    man = json.loads((adir / "manifest.json").read_text(encoding="utf-8"))
    store = Store(Path(args.ckpt))

    entries = man["blocks"]
    if args.blocks != "all":
        want = {int(b) for b in args.blocks.split(",")}
        entries = [e for e in entries if e["block"] in want]

    fails: list[str] = []
    per_gemm = []
    fp16_err = []
    int8_bytes = 0

    for e in entries:
        region = (adir / e["region"]).read_bytes()
        scales_raw = (adir / e["scales"]).read_bytes()
        fp16 = (adir / e["fp16"]).read_bytes()

        # 1. sizes + checksums + alignment
        if len(region) != e["region_bytes"]:
            fails.append(f"block {e['block']}: region size {len(region)} != manifest {e['region_bytes']}")
        if sha256_file(adir / e["region"]) != e["region_sha256"]:
            fails.append(f"block {e['block']}: region sha256 mismatch")
        if len(scales_raw) != e["scales_bytes"]:
            fails.append(f"block {e['block']}: scales size {len(scales_raw)} != manifest {e['scales_bytes']}")
        if sha256_file(adir / e["scales"]) != e["scales_sha256"]:
            fails.append(f"block {e['block']}: scales sha256 mismatch")
        if sha256_file(adir / e["fp16"]) != e["fp16_sha256"]:
            fails.append(f"block {e['block']}: fp16 sha256 mismatch")
        if e["region_bytes"] / 2**20 > GATE1_REGION_MIB:
            fails.append(f"block {e['block']}: region {e['region_bytes']/2**20:.2f} MiB > {GATE1_REGION_MIB}")
        for g in e["gemms"]:
            if g["offset"] % ALIGN:
                fails.append(f"block {e['block']} {g['name']}: offset {g['offset']} not {ALIGN}-aligned")
            if g["offset"] + g["nbytes"] > len(region):
                fails.append(f"block {e['block']} {g['name']}: runs past the region end")

        int8_bytes += e["region_bytes"]
        scales = np.frombuffer(scales_raw, dtype="<f4")

        for g in e["gemms"]:
            n_out, k = g["shape"]
            nbytes = n_out * k
            if nbytes != g["nbytes"]:
                fails.append(f"block {e['block']} {g['name']}: shape says {nbytes} B, manifest {g['nbytes']} B")
            q_disk = np.frombuffer(region, dtype=np.int8, count=nbytes,
                                   offset=g["offset"]).reshape(n_out, k)
            s = scales[g["scale_offset"] // 4: g["scale_offset"] // 4 + n_out]
            if s.size != n_out:
                fails.append(f"block {e['block']} {g['name']}: {s.size} scales for {n_out} channels")
                continue

            W = store.get(g["full_name"]).astype(np.float32)

            # 2. re-quantise the source and demand byte identity
            q_ref, s_ref, _ = quant_per_output_channel(W)
            if not np.array_equal(q_disk, q_ref):
                bad = int((q_disk != q_ref).sum())
                fails.append(f"block {e['block']} {g['name']}: {bad} INT8 bytes differ from a fresh quantisation")
            if not np.array_equal(s.astype(np.float32), s_ref):
                fails.append(f"block {e['block']} {g['name']}: scales differ from a fresh quantisation")

            # 3. dequant vs W from the DISK bytes
            Wd = q_disk.astype(np.float32) * s[:, None]
            W64 = W.astype(np.float64)
            d = W64 - Wd.astype(np.float64)
            nref = float(np.linalg.norm(W64))
            nd = float(np.linalg.norm(Wd.astype(np.float64)))
            rec = {
                "block": e["block"], "gemm": g["name"], "shape": [n_out, k],
                "macs": n_out * k,
                "rel_fro": float(np.linalg.norm(d) / nref) if nref else 0.0,
                "cosine": float((W64 * Wd).sum() / (nref * nd)) if nref and nd else 1.0,
                "max_abs_err": float(np.abs(d).max()),
                "saturated_frac": float((np.abs(q_disk) == CLIP).mean()),
                "scale_spread": float(s.max() / s.min()),
            }
            per_gemm.append(rec)
            if rec["cosine"] < 0.999:
                fails.append(f"block {e['block']} {g['name']}: cosine {rec['cosine']:.6f} < 0.999")
            del W, Wd, q_disk, q_ref

        # 4. FP16 remainder error (bias / norm / AdaLN / io weights): report, do not fail on
        #    it — the locked design keeps these FP16 on purpose.
        for t in e["fp16_tensors"]:
            if t["offset"] % ALIGN:
                fails.append(f"block {e['block']} {t['name']}: fp16 offset {t['offset']} not {ALIGN}-aligned")
            if t["offset"] + t["nbytes"] > len(fp16):
                fails.append(f"block {e['block']} {t['name']}: fp16 runs past the file end")
            W = store.get(t["name"]).astype(np.float32)
            n = int(W.size)
            W16 = np.frombuffer(fp16, dtype="<f2", count=n, offset=t["offset"]).astype(np.float32).reshape(W.shape)
            nref = float(np.linalg.norm(W.astype(np.float64)))
            d = W.astype(np.float64) - W16.astype(np.float64)
            fp16_err.append({"block": e["block"], "name": t["name"],
                             "rel_fro": float(np.linalg.norm(d) / nref) if nref else 0.0})
            del W, W16

    # Top-level FP16 file (patch embedding, io weights, cond embedder, proj_out, top AdaLN):
    # same error check, plus a checksum against the manifest.
    top_meta = man.get("top_level_fp16")
    if top_meta:
        top_file = adir / top_meta["path"]
        if sha256_file(top_file) != top_meta["sha256"]:
            fails.append("top.fp16.bin sha256 mismatch")
        top_buf = top_file.read_bytes()
        if len(top_buf) != top_meta["bytes"]:
            fails.append(f"top.fp16.bin size {len(top_buf)} != manifest {top_meta['bytes']}")
        for t in top_meta["tensors"]:
            if t["offset"] % ALIGN:
                fails.append(f"top {t['name']}: offset {t['offset']} not {ALIGN}-aligned")
            W = store.get(t["name"]).astype(np.float32)
            n = int(W.size)
            if t["offset"] + n * 2 > len(top_buf):
                fails.append(f"top {t['name']}: runs past the file end")
                continue
            W16 = np.frombuffer(top_buf, dtype="<f2", count=n, offset=t["offset"]).astype(np.float32).reshape(W.shape)
            nref = float(np.linalg.norm(W.astype(np.float64)))
            d = W.astype(np.float64) - W16.astype(np.float64)
            fp16_err.append({"block": -1, "name": t["name"],
                             "rel_fro": float(np.linalg.norm(d) / nref) if nref else 0.0})
            del W, W16

    mask = {e["block"] for e in entries}
    top_bytes = man.get("top_level_fp16", {}).get("bytes", 0)
    int8_bytes_total = man["totals"]["int8_bytes"] if len(entries) == len(man["blocks"]) else int8_bytes

    # Coverage: the manifest must account for every checkpoint tensor exactly once,
    # with the expert-side tensors explicitly deferred (deliverable 2). A verifier that
    # only checks what the manifest lists cannot notice what the manifest forgot.
    listed = set()
    for e in man["blocks"]:
        listed.update(g["full_name"] for g in e["gemms"])
        listed.update(t["name"] for t in e["fp16_tensors"])
    listed.update(t["name"] for t in man.get("top_level_fp16", {}).get("tensors", []))
    unknown = sorted(listed - set(store.hdr))
    if unknown:
        fails.append(f"manifest lists {len(unknown)} tensors absent from the checkpoint: {unknown[:3]}")
    missing = sorted(set(store.hdr) - listed)
    deferred = [n for n in missing if "lora" in n or "condition_embedder_lora" in n
                or n.startswith("norm_out_lora") or n.startswith("proj_out_lora")]
    truly_missing = [n for n in missing if n not in deferred]
    if truly_missing:
        fails.append(f"{len(truly_missing)} checkpoint tensors are in no file: {truly_missing[:5]}")

    def mac_weighted(key: str) -> float:
        num = sum(r["macs"] * r[key] for r in per_gemm)
        den = sum(r["macs"] for r in per_gemm)
        return num / den if den else 0.0

    worst = max(per_gemm, key=lambda r: r["rel_fro"]) if per_gemm else None
    mincos = min(per_gemm, key=lambda r: r["cosine"]) if per_gemm else None
    max_sat = max((r["saturated_frac"] for r in per_gemm), default=0.0)

    verdict = {
        "blocks_verified": sorted(mask),
        "gemms_verified": len(per_gemm),
        "int8_bytes": int8_bytes_total,
        "int8_delta_bytes": int8_bytes_total - GATE1_INT8_BYTES,
        "gate1_bytes_ok": abs(int8_bytes_total - GATE1_INT8_BYTES) <= 2**20,
        "max_region_mib": max((e["region_mib"] for e in entries), default=0.0),
        "gate1_region_ok": max((e["region_mib"] for e in entries), default=0.0) <= GATE1_REGION_MIB,
        "mac_weighted_rel_fro": mac_weighted("rel_fro"),
        "mac_weighted_cosine": mac_weighted("cosine"),
        "worst_gemm": worst,
        "min_cosine_gemm": mincos,
        "gate1_cosine_ok": (mincos["cosine"] >= 0.999) if mincos else False,
        "max_saturated_frac": max_sat,
        "fp16_remainder_max_rel_fro": max((f["rel_fro"] for f in fp16_err), default=0.0),
        "fp16_remainder_tensors": len(fp16_err),
        "coverage_missing_from_files": len(truly_missing),
        "coverage_deferred_expert": len(deferred),
        "checksum_failures": [f for f in fails if "sha256" in f],
        "fails": fails,
        "verdict": "PASS" if not fails else "FAIL",
    }

    print(f"blocks verified        : {len(mask)}   GEMMs: {len(per_gemm)}")
    print(f"INT8 bytes             : {int8_bytes_total:,}  (delta {verdict['int8_delta_bytes']:+,} B) "
          f"{'PASS' if verdict['gate1_bytes_ok'] else 'FAIL'}")
    print(f"max region             : {verdict['max_region_mib']:.2f} MiB "
          f"{'PASS' if verdict['gate1_region_ok'] else 'FAIL'} (limit {GATE1_REGION_MIB})")
    print(f"MAC-weighted GEMM err  : {verdict['mac_weighted_rel_fro']*100:.2f}% rel. Frobenius, "
          f"cosine {verdict['mac_weighted_cosine']:.6f}")
    if worst:
        print(f"worst GEMM             : block {worst['block']} {worst['gemm']} "
              f"{worst['shape']}  {worst['rel_fro']*100:.2f}%  cos {worst['cosine']:.6f}")
        print(f"min-cosine GEMM        : block {mincos['block']} {mincos['gemm']} "
              f"cos {mincos['cosine']:.6f}  {'PASS' if verdict['gate1_cosine_ok'] else 'FAIL'} vs 0.999")
    print(f"saturation (|q|=127)   : max {max_sat*100:.3f}% of a GEMM's weights")
    print(f"FP16 remainder err     : max {verdict['fp16_remainder_max_rel_fro']*100:.3f}% rel. Frobenius "
          f"({len(fp16_err)} tensors; bias/norm/AdaLN/io weights, FP16 by design)")
    print(f"coverage               : {verdict['coverage_missing_from_files']} checkpoint tensors in no file, "
          f"{verdict['coverage_deferred_expert']} expert-side tensors deferred to deliverable 2")
    print()
    if fails:
        print(f"VERDICT: FAIL ({len(fails)} problem(s))")
        for f in fails[:20]:
            print("  -", f)
    else:
        print("VERDICT: PASS — every INT8 byte on disk equals a fresh quantisation of the "
              "checkpoint, sizes and checksums match the manifest.")

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"kind": "gate1_artifact_verify", "artifact": str(adir),
                                 **verdict, "per_gemm": per_gemm, "fp16": fp16_err}, indent=2),
                     encoding="utf-8")
        print(f"wrote {p}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
