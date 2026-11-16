#!/usr/bin/env python3
"""Bake the measured per-layer activation scales into the VAE artifact.

`calibrate.py` (P3.9) produced `max|x|/127` for every one of the decoder's 36
layers on the tensors the INT8 taps actually consume:
`artifacts/calib/act_scales_vae_*.json`. `npu_block convseq` was still pinning
`VAE_SA = 1/127` for all 36, which clips 34 of them (median 4.94x, worst 32.06x
on `up_blocks.1.resnets.0.conv_shortcut`) — the measured reason the decode
"executes but does not decode".

This tool writes the missing device input:

  artifacts/vae/act_scales_f32.bin   36 x float32, artifact layer order
  artifacts/vae/vae_act_scales.json  provenance + per-layer values + ratios

The table is *derived*, never typed: the layer order comes from
`artifacts/vae/vae_int8.json` (the artifact the device executes), the values from
the newest calibration JSON, and the mapping is asserted (names must match
element-wise, 36/36, every SA finite and > 0). A re-read verifier re-parses the
written bytes and demands float identity with the source JSON, so a byte-order or
stride mistake is a hard failure instead of a silently wrong decode.

Float32 storage is exact for these values to ~1e-7 relative; the verifier
compares the *float32* round-trip of each value, not the float64 source, and
prints the worst delta.

    python phase1_quant/vae_sa.py                       # newest calib file
    python phase1_quant/vae_sa.py --calib artifacts/calib/act_scales_vae_20261116_054009.json
    python phase1_quant/vae_sa.py --check               # verify, write nothing
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VAE_DIR = REPO / "artifacts" / "vae"
CALIB_DIR = REPO / "artifacts" / "calib"
ART_JSON = VAE_DIR / "vae_int8.json"
OUT_BIN = VAE_DIR / "act_scales_f32.bin"
OUT_JSON = VAE_DIR / "vae_act_scales.json"
NOMINAL = 1.0 / 127.0


def f32(x: float) -> float:
    """The value the device will actually see (round-trip through float32)."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def newest_calib() -> Path:
    files = sorted(CALIB_DIR.glob("act_scales_vae_*.json"))
    if not files:
        raise SystemExit(f"no act_scales_vae_*.json under {CALIB_DIR} — run calibrate.py first")
    return files[-1]


def load() -> tuple[list[dict], dict, Path]:
    art = json.loads(ART_JSON.read_text(encoding="utf-8"))
    calib_path = CALIB
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    a_names = [L["name"] for L in art["layers"]]
    c_names = [L["layer"] for L in calib["layers"]]
    if a_names != c_names:
        only_a = [n for n in a_names if n not in set(c_names)]
        only_c = [n for n in c_names if n not in set(a_names)]
        raise SystemExit(
            "layer order mismatch between the artifact and the calibration:\n"
            f"  artifact-only ({len(only_a)}): {only_a[:6]}\n"
            f"  calib-only    ({len(only_c)}): {only_c[:6]}\n"
            "The SA table is indexed by layer position on the device, so this is fatal.")
    rows = []
    for i, (L, C) in enumerate(zip(art["layers"], calib["layers"])):
        sa = float(C["sa"])
        if not (sa > 0.0) or sa != sa or sa in (float("inf"), float("-inf")):
            raise SystemExit(f"layer {i} {L['name']}: non-finite/non-positive SA {C['sa']}")
        rows.append(dict(idx=i, name=L["name"], sa=f32(sa), sa_src=sa, max_abs=C["max_abs"],
                         ratio_vs_nominal=f32(sa) / NOMINAL,
                         needs_larger=(f32(sa) > NOMINAL)))
    return rows, calib, calib_path


def write(rows: list[dict]) -> tuple[str, Path]:
    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)
    blob = b"".join(struct.pack("<f", r["sa"]) for r in rows)
    OUT_BIN.write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    doc = dict(
        built=dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        source_calib=str(CALIB.relative_to(REPO)).replace("\\", "/"),
        source_calib_sha=hashlib.sha256(CALIB.read_bytes()).hexdigest()[:16],
        artifact=str(ART_JSON.relative_to(REPO)).replace("\\", "/"),
        artifact_sha=hashlib.sha256(ART_JSON.read_bytes()).hexdigest()[:16],
        file=dict(name=OUT_BIN.name, bytes=len(blob), sha256=sha,
                  layout="float32[NLAYER], artifact layer order, one activation scale per layer "
                         "(the input tensor's max|x|/127 measured on the real decode)"),
        nominal=NOMINAL,
        n_layers=len(rows),
        n_larger_than_nominal=sum(1 for r in rows if r["needs_larger"]),
        ratios=dict(
            min=min(r["ratio_vs_nominal"] for r in rows),
            median=sorted(r["ratio_vs_nominal"] for r in rows)[len(rows) // 2],
            max=max(r["ratio_vs_nominal"] for r in rows)),
        layers=rows,
        note=("SA is the activation scale the device must quantise this layer's INPUT with. "
              "It is per layer, not per GEMM: the decode's tensors are per layer. Output tensors "
              "are requantised to the consumer layer's SA, so the byte-copy wiring stays a copy."),
    )
    OUT_JSON.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return sha, OUT_JSON


def verify(rows: list[dict]) -> int:
    """Re-read the bytes and demand float identity with the source JSON."""
    blob = OUT_BIN.read_bytes()
    if len(blob) != 4 * len(rows):
        print(f"FAIL size: {len(blob)} B != {4 * len(rows)} B")
        return 1
    worst = 0.0
    for i, r in enumerate(rows):
        got = struct.unpack_from("<f", blob, 4 * i)[0]
        want = f32(r["sa_src"])
        worst = max(worst, abs(got - want))
        if got != want:
            print(f"FAIL layer {i} {r['name']}: {got!r} != {want!r}")
            return 1
    ratios = [r["ratio_vs_nominal"] for r in rows]
    larger = sum(1 for x in ratios if x > 1.0)
    print(f"PASS act_scales_f32.bin: {len(rows)} layers, {len(blob)} B, float identity "
          f"(worst delta {worst:.3g})")
    print(f"     vs nominal 1/127: {larger}/{len(rows)} layers need a LARGER scale, "
          f"{len(rows) - larger} smaller; ratios {min(ratios):.3f} .. {max(ratios):.3f} "
          f"(median {sorted(ratios)[len(ratios) // 2]:.3f})")
    print(f"     SA {min(r['sa'] for r in rows):.6g} .. {max(r['sa'] for r in rows):.6g}")
    return 0


def main() -> int:
    global CALIB
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="", help="calibration JSON (default: newest vae one)")
    ap.add_argument("--check", action="store_true", help="verify the existing table, write nothing")
    args = ap.parse_args()
    CALIB = Path(args.calib) if args.calib else newest_calib()
    if not CALIB.is_absolute():
        CALIB = REPO / CALIB
    rows, calib, calib_path = load()
    print(f"calibration: {calib_path.relative_to(REPO)} ({calib.get('built', 'built?')}), "
          f"{len(rows)} layers, {len(rows)}/36 matched to {ART_JSON.name} by name and position")
    if not args.check:
        sha, out = write(rows)
        print(f"wrote {OUT_BIN.relative_to(REPO)} ({len(rows) * 4} B, sha256 {sha[:16]})")
        print(f"wrote {out.relative_to(REPO)}")
    return verify(rows)


if __name__ == "__main__":
    sys.exit(main())
