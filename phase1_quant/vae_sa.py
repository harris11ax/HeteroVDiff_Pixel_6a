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
OUT2_BIN = VAE_DIR / "act_scales2_f32.bin"
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
        # P3.11 (2026-11-16): the OUTPUT range. Absent in a pre-P3.11 calibration,
        # where `max_abs_out` was the whole decode's output (a flat 1.0) — refuse
        # rather than derive a table from a number that was never an output range.
        out = C.get("sa_out")
        if out is None:
            raise SystemExit(
                f"layer {i} {L['name']}: calibration has no `sa_out` — it predates the "
                "per-layer OUTPUT range (P3.11). Re-run `phase1_quant/calibrate.py --target vae`.")
        out = float(out)
        if not (out > 0.0) or out != out:
            raise SystemExit(f"layer {i} {L['name']}: non-finite/non-positive sa_out {out}")
        rows.append(dict(idx=i, name=L["name"], sa=f32(sa), sa_src=sa, max_abs=C["max_abs"],
                         sa_out=f32(out), sa_out_src=out, max_abs_out=C["max_abs_out"],
                         out_over_in=f32(out) / f32(sa),
                         ratio_vs_nominal=f32(sa) / NOMINAL,
                         needs_larger=(f32(sa) > NOMINAL)))
    return rows, calib, calib_path


def write(rows: list[dict]) -> tuple[str, Path]:
    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)
    # (1) the legacy 1 x 36 view: SA_in only, what the currently-built `convseq`
    #     binary reads. Kept so a device binary that predates P3.11 still runs.
    blob = b"".join(struct.pack("<f", r["sa"]) for r in rows)
    OUT_BIN.write_bytes(blob)
    # (2) P3.11: 2 x 36 = SA_in[36] ++ SA_out[36]. The output block is what a
    #     layer's graph must requantise INTO, so the layer's own output stops
    #     being clipped to a consumer's input range that may be 89x smaller
    #     (measured max out/in = 15.5 within a layer, 89x across a norm).
    blob2 = (b"".join(struct.pack("<f", r["sa"]) for r in rows) +
             b"".join(struct.pack("<f", r["sa_out"]) for r in rows))
    OUT2_BIN.write_bytes(blob2)
    sha = hashlib.sha256(blob2).hexdigest()
    oi = [r["out_over_in"] for r in rows]
    doc = dict(
        built=dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        source_calib=str(CALIB.relative_to(REPO)).replace("\\", "/"),
        source_calib_sha=hashlib.sha256(CALIB.read_bytes()).hexdigest()[:16],
        artifact=str(ART_JSON.relative_to(REPO)).replace("\\", "/"),
        artifact_sha=hashlib.sha256(ART_JSON.read_bytes()).hexdigest()[:16],
        file=dict(name=OUT2_BIN.name, bytes=len(blob2), sha256=sha,
                  layout="float32[2*NLAYER] = SA_in[NLAYER] ++ SA_out[NLAYER], artifact "
                         "layer order; SA_in = the input tensor's max|x|/127, SA_out = the "
                         "OUTPUT tensor's max|x|/127, both measured on the real decode"),
        legacy_file=dict(name=OUT_BIN.name, bytes=len(blob),
                         note="SA_in[NLAYER] only, for a convseq built before P3.11"),
        nominal=NOMINAL,
        n_layers=len(rows),
        n_larger_than_nominal=sum(1 for r in rows if r["needs_larger"]),
        ratios=dict(
            min=min(r["ratio_vs_nominal"] for r in rows),
            median=sorted(r["ratio_vs_nominal"] for r in rows)[len(rows) // 2],
            max=max(r["ratio_vs_nominal"] for r in rows)),
        out_over_in=dict(min=min(oi), median=sorted(oi)[len(oi) // 2], max=max(oi)),
        layers=rows,
        note=("SA_in is the activation scale the device must quantise this layer's INPUT with; "
              "SA_out is the scale its OUTPUT is measured to live in. They are NOT equal and "
              "the ratio is not a constant (0.27x .. 15.5x here) because the decoder puts a "
              "WanRMS_norm + SiLU between conv1 and conv2 (F.normalize over channels, times "
              "sqrt(C), times gamma), which renormalises the tensor per pixel. So the wiring "
              "step between two layers is a rescale by SA_out[i]/SA_in[i+1] at best, and a "
              "norm+SiLU+requant at worst — it is never a pure byte copy where a norm sits "
              "between them."),
    )
    OUT_JSON.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return sha, OUT_JSON


def verify(rows: list[dict]) -> int:
    """Re-read the bytes and demand float identity with the source JSON."""
    ok = 0
    for path, sel, label in ((OUT2_BIN, ("sa", "sa_out"), "act_scales2_f32.bin"),
                             (OUT_BIN, ("sa",), "act_scales_f32.bin")):
        blob = path.read_bytes()
        n = len(sel) * len(rows)
        if len(blob) != 4 * n:
            print(f"FAIL {label} size: {len(blob)} B != {4 * n} B")
            return 1
        worst = 0.0
        for b, key in enumerate(sel):
            for i, r in enumerate(rows):
                got = struct.unpack_from("<f", blob, 4 * (b * len(rows) + i))[0]
                want = f32(r[key + "_src"])
                worst = max(worst, abs(got - want))
                if got != want:
                    print(f"FAIL {label} block {key} layer {i} {r['name']}: {got!r} != {want!r}")
                    return 1
        print(f"PASS {label}: {len(sel)} x {len(rows)} layers, {len(blob)} B, float identity "
              f"(worst delta {worst:.3g})")
        ok += 1
    if ok != 2:
        return 1
    ratios = [r["ratio_vs_nominal"] for r in rows]
    larger = sum(1 for x in ratios if x > 1.0)
    oi = [r["out_over_in"] for r in rows]
    print(f"     SA_in vs nominal 1/127: {larger}/{len(rows)} layers need a LARGER scale, "
          f"ratios {min(ratios):.3f} .. {max(ratios):.3f} (median {sorted(ratios)[len(ratios) // 2]:.3f})")
    print(f"     SA_in {min(r['sa'] for r in rows):.6g} .. {max(r['sa'] for r in rows):.6g}   "
          f"SA_out {min(r['sa_out'] for r in rows):.6g} .. {max(r['sa_out'] for r in rows):.6g}")
    print(f"     SA_out/SA_in: {min(oi):.3f} .. {max(oi):.3f} (median {sorted(oi)[len(oi) // 2]:.3f})")
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
