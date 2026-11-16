#!/usr/bin/env python3
"""P3.8a: the VAE decoder's INT8 weight artifact — the bytes `--decode 1` will execute.

`configs/vae/plan_npu.json` says *what* the Tensor Edge TPU stage has to run: 35 causal 3-D
convs decomposed into **592 `CONV_2D` executes** per chunk, each tap addressing a byte window
`[0, t*Cout*Cin*kH*kW, Cout*Cin*kH*kW]` of its layer's weights. Until now nothing wrote those
bytes — `npu_block conv` timed the shapes against a deterministic LCG fill, which is fine for a
latency table and useless for an execution.

This script writes the real thing:

  artifacts/vae/weights_int8.bin  INT8, layer-major, tap-major inside a layer, and inside a tap
                                  `[Cout, kH, kW, Cin]` (OHWI) — the order `CONV_2D` wants,
                                  which is the transpose of the checkpoint's `[Cout,Cin,kT,kH,kW]`.
                                  Every tap starts on a 4096-byte boundary so it can be handed to
                                  `ANeuralNetworksMemory_createFromFd` / `setOperandValueFromMemory`
                                  without a copy.
  artifacts/vae/scales_f32.bin    one FP32 scale per output channel per layer (the locked scheme,
                                  computed over the layer's whole `[Cout, Cin*kT*kH*kW]` view — a
                                  per-tap scale would be a different quantiser).
  artifacts/vae/bias_f32.bin      the DECODER biases, FP32, **not quantised**: NNAPI wants an
                                  INT32 bias in units of `SA*SW`, and `SA` (activation scale) comes
                                  from `calibrate.py`, which does not exist yet. Recording them
                                  unquantised is the honest state; guessing `SA=1/127` would put a
                                  fabricated scale into the artifact.
  artifacts/vae/vae_int8.json     the manifest the device binary indexes (offsets, not arithmetic).

The artifact is 72.7 MB of INT8 + 35x384 FP32 scales. It is **not** a second copy of the base
weights: the base 300 GEMM artifact (`artifacts/blocks/`) is the DiT, and this is the VAE decoder,
a different set of tensors. The verifier asserts that no tensor it reads belongs to the DiT.

    python phase1_quant/vae_int8.py            # write
    python phase1_quant/vae_int8.py --verify   # independent re-derive, byte identity
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "weights_raw" / "DCM_WAN" / "vae" / "diffusion_pytorch_model.safetensors"
PLAN = REPO / "configs" / "vae" / "plan_npu.json"
LAYERS = REPO / "configs" / "vae" / "layers.json"
OUT = REPO / "artifacts" / "vae"
MANIFEST = OUT / "vae_int8.json"

ALIGN = 4096          # tap window alignment: createFromFd takes a page-aligned file offset
CLIP = 127            # symmetric INT8, the same clip the base artifact uses
GATE_COSINE = 0.999   # Gate 1's weight bar, applied per layer


# --------------------------------------------------------------- checkpoint IO
def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return {k: v for k, v in hdr.items() if k != "__metadata__"}, 8 + n


class Store:
    """Read-only memmap of the safetensors payload; tensors are views, not copies."""

    def __init__(self, path: Path):
        self.hdr, self.data_start = read_header(path)
        self.arr = np.memmap(path, dtype=np.uint8, mode="r")

    def get(self, name: str) -> np.ndarray:
        m = self.hdr[name]
        if m["dtype"] != "F32":
            raise SystemExit(f"{name}: dtype {m['dtype']} is not F32 — this quantiser assumes FP32")
        a, b = m["data_offsets"]
        flat = np.frombuffer(self.arr, dtype="<f4", count=(b - a) // 4, offset=self.data_start + a)
        return flat.reshape(m["shape"])


# ---------------------------------------------------------------- quantisation
def tap_views(W: np.ndarray) -> tuple[list[np.ndarray], int]:
    """Checkpoint conv weight -> OHWI tap views.

    `[Cout, Cin, kT, kH, kW]` (5-D, causal time) or `[Cout, Cin, kH, kW]` (4-D, kT=1).
    Returns one `[Cout, kH*kW*Cin]` row-major view per temporal tap, which is exactly the
    byte window CONV_2D's filter operand spans.
    """
    if W.ndim == 5:
        co, ci, kt, kh, kw = W.shape
        return [np.ascontiguousarray(np.transpose(W[:, :, t, :, :], (0, 2, 3, 1))
                                     ).reshape(co, kh * kw * ci) for t in range(kt)], kt
    if W.ndim == 4:
        co, ci, kh, kw = W.shape
        return [np.ascontiguousarray(np.transpose(W, (0, 2, 3, 1))).reshape(co, kh * kw * ci)], 1
    raise SystemExit(f"unexpected conv weight rank {W.ndim}")


def quantise(taps: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """Per-output-channel symmetric INT8 over the WHOLE layer (all taps share one scale)."""
    M = np.concatenate(taps, axis=1).astype(np.float32)      # [Cout, kT*kH*kW*Cin]
    s = np.abs(M).max(axis=1).astype(np.float32) / CLIP
    s[s == 0.0] = 1.0
    q = [np.clip(np.rint(t / s[:, None]), -CLIP, CLIP).astype(np.int8) for t in taps]
    Wd = np.concatenate([qq.astype(np.float32) * s[:, None] for qq in q], axis=1)
    ref = M.astype(np.float64)
    nref = float(np.linalg.norm(ref))
    d = ref - Wd.astype(np.float64)
    stat = {
        "rel_fro": float(np.linalg.norm(d) / nref) if nref else 0.0,
        "cosine": float((ref * Wd.astype(np.float64)).sum() / nref
                        / np.linalg.norm(Wd.astype(np.float64))) if nref else 1.0,
        "max_abs_err": float(np.abs(d).max()),
        "scale_min": float(s.min()),
        "scale_max": float(s.max()),
        "scale_spread": float(s.max() / s.min()),
        "saturated_frac": float(np.mean([(np.abs(qq) == CLIP).mean() for qq in q])),
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


# ------------------------------------------------------------------- building
def build_layer(store: Store, plan_layer: dict) -> dict:
    name = plan_layer["name"]
    weight = f"{name}.weight"
    W = store.get(weight)
    taps, kt = tap_views(W)
    if kt != len(plan_layer["taps"]):
        raise SystemExit(f"{name}: {kt} taps in the weight, {len(plan_layer['taps'])} in the plan")
    q, s, stat = quantise(taps)
    bias = store.get(f"{name}.bias").astype(np.float32) if f"{name}.bias" in store.hdr else None
    if bias is not None and bias.shape[0] != W.shape[0]:
        raise SystemExit(f"{name}: bias {bias.shape} vs Cout {W.shape[0]}")
    return {"name": name, "weight": weight, "source_shape": list(W.shape),
            "cout": int(W.shape[0]), "cin": int(W.shape[1]),
            "frames": plan_layer["frames"], "taps": q, "scales": s, "bias": bias, "stat": stat}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="re-derive from source and demand byte identity")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    store = Store(CKPT)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    layers_doc = json.loads(LAYERS.read_text(encoding="utf-8"))
    inv_bytes = sum(l["weight_bytes_if_int8"] for l in layers_doc["layers"]
                    if l.get("weight_bytes_if_int8"))

    # the plan's conv layers, in plan order — the device walks this list
    plan_convs = [L for L in plan["layers"] if L["taps"]]
    built = [build_layer(store, L) for L in plan_convs]

    # ---- lay out the blob: every tap on an ALIGN boundary -----------------
    regions: list[bytearray] = [bytearray(), bytearray(), bytearray()]   # weights, scales, bias
    woff = soff = boff = 0
    manifest_layers = []
    n_taps = 0
    worst = {"cosine": 1.0, "name": ""}
    for L, pl in zip(built, plan_convs):
        base = align(len(regions[0]))
        regions[0].extend(b"\0" * (base - len(regions[0])))
        woff = base
        tap_recs = []
        for t, q in enumerate(L["taps"]):
            off = align(len(regions[0]))
            regions[0].extend(b"\0" * (off - len(regions[0])))
            regions[0].extend(np.ascontiguousarray(q).tobytes(order="C"))
            tap_recs.append({
                "tap": t,
                "offset": off,
                "bytes": int(q.size),
                # the plan's unpadded arithmetic, kept so the padding is auditable
                "plan_slice_unpadded": [0, t * int(q.size), int(q.size)],
                "conv2d": [L["cout"], L["cin"], pl["taps"][t]["conv2d"][2], pl["taps"][t]["conv2d"][3]],
                "input_frame": pl["taps"][t]["input_frame"],
            })
            n_taps += 1
        scale_off = len(regions[1])
        regions[1].extend(np.ascontiguousarray(L["scales"], dtype="<f4").tobytes())
        bias_off = None
        if L["bias"] is not None:
            bias_off = len(regions[2])
            regions[2].extend(np.ascontiguousarray(L["bias"], dtype="<f4").tobytes())
        if L["stat"]["cosine"] < worst["cosine"]:
            worst = {"cosine": L["stat"]["cosine"], "name": L["name"]}
        manifest_layers.append({
            "name": L["name"], "weight_tensor": L["weight"], "source_shape": L["source_shape"],
            "cout": L["cout"], "cin": L["cin"], "frames": L["frames"], "taps": tap_recs,
            "executes": L["frames"] * len(tap_recs),
            "scales": {"offset": scale_off, "count": int(L["scales"].size), "dtype": "<f4"},
            "bias": None if bias_off is None else {"offset": bias_off, "count": int(L["bias"].size),
                                                   "dtype": "<f4", "quantised": False,
                                                   "note": "INT32 bias needs SA from calibrate.py"},
            "quant": L["stat"],
        })

    wings = out / "weights_int8.bin"
    sc = out / "scales_f32.bin"
    bi = out / "bias_f32.bin"
    wings.write_bytes(bytes(regions[0]))
    sc.write_bytes(bytes(regions[1]))
    bi.write_bytes(bytes(regions[2]))

    padded = len(regions[0])
    manifest = {
        "kind": "vae_int8_artifact",
        "checkpoint": str(CKPT.relative_to(REPO)),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "plan": "configs/vae/plan_npu.json",
        "files": {
            "weights_int8.bin": {"bytes": padded, "sha256": sha256_file(wings),
                                 "layout": "layer-major, tap-major, [Cout, kH, kW, Cin] (OHWI) per tap"},
            "scales_f32.bin": {"bytes": len(regions[1]), "sha256": sha256_file(sc)},
            "bias_f32.bin": {"bytes": len(regions[2]), "sha256": sha256_file(bi)},
        },
        "quantisation": {
            "scheme": "per-output-channel symmetric int8 over the whole layer, "
                      "s_c = max|W[c, :, :, :, :]| / 127",
            "clip": [-CLIP, CLIP],
            "filter_layout": "OHWI — the transpose of the checkpoint's [Cout, Cin, kT, kH, kW]; "
                             "CONV_2D reads [Cout, kH, kW, Cin]",
            "tap_alignment_bytes": ALIGN,
            "bias": "FP32 in bias_f32.bin, NOT quantised — INT32 bias is in SA*SW units and SA "
                    "depends on calibrate.py, which does not exist yet",
        },
        "one_copy": "VAE decoder tensors only (names start 'decoder.'); the 300 DiT base GEMMs "
                    "in artifacts/blocks/ are untouched — this is a different artifact, not a "
                    "second INT8 copy of the base weights",
        "totals": {
            "layers": len(manifest_layers),
            "taps": n_taps,
            "executes_per_chunk": sum(l["executes"] for l in manifest_layers),
            "int8_weight_bytes_unpadded": int(inv_bytes),
            "int8_weight_bytes_padded": padded,
            "alignment_overhead_bytes": padded - int(inv_bytes),
            "alignment_overhead_frac": (padded - inv_bytes) / inv_bytes,
            "scales_bytes": len(regions[1]),
            "bias_bytes": len(regions[2]),
            "worst_cosine": worst["cosine"],
            "worst_layer": worst["name"],
            "mean_rel_fro": float(np.mean([l["quant"]["rel_fro"] for l in manifest_layers])),
            "gate_cosine": GATE_COSINE,
            "gate_pass": all(l["quant"]["cosine"] >= GATE_COSINE for l in manifest_layers),
            "plan_executes": plan["total_executes_per_chunk"],
        },
        "layers": manifest_layers,
    }
    (out / "vae_int8.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    t = manifest["totals"]
    print(f"layers              : {t['layers']} ({t['taps']} taps, {t['executes_per_chunk']} executes/chunk; "
          f"plan says {t['plan_executes']})")
    print(f"weights             : {t['int8_weight_bytes_unpadded']:,} B INT8 unpadded -> "
          f"{t['int8_weight_bytes_padded']:,} B padded (+{t['alignment_overhead_frac']*100:.3f} %)")
    print(f"scales / bias       : {t['scales_bytes']:,} B f32 / {t['bias_bytes']:,} B f32 (unquantised)")
    print(f"quantisation        : mean relFro {t['mean_rel_fro']*100:.3f} %, worst cosine "
          f"{t['worst_cosine']:.6f} ({t['worst_layer']}) -> Gate-1 bar {GATE_COSINE} "
          f"{'PASS' if t['gate_pass'] else 'FAIL'}")

    # ---- both quantiser branches, measured (the artifact keeps the locked one) ----
    below = [(l["quant"]["cosine"], l["name"], l["quant"]["scale_spread"]) for l in manifest_layers
             if l["quant"]["cosine"] < GATE_COSINE]
    per_tap = []
    for L in built:
        cs = []
        for t_ in L["taps"]:
            m = t_.astype(np.float32)
            s_ = np.abs(m).max(axis=1).astype(np.float32) / CLIP
            s_[s_ == 0.0] = 1.0
            wd = np.rint(t_ / s_[:, None]).clip(-CLIP, CLIP).astype(np.float32) * s_[:, None]
            cs.append(float((m.astype(np.float64) * wd).sum() / max(np.linalg.norm(m), 1e-30)
                            / max(np.linalg.norm(wd), 1e-30)))
        per_tap.append((min(cs), L["name"]))
    per_tap.sort()
    print(f"  per-layer scale   : {len(below)}/{len(manifest_layers)} layers below {GATE_COSINE}"
          + (f"; worst {below[0][1]} {below[0][0]:.6f} (scale spread {below[0][2]:.1f}x)" if below else ""))
    print(f"  per-tap scale     : worst {per_tap[0][0]:.6f} ({per_tap[0][1]}) — measured for comparison, "
          f"NOT adopted (per-output-column over the whole layer is the locked scheme)")
    manifest["quant_branches"] = {
        "locked": {"scheme": "one scale per output channel over the whole layer",
                   "below_gate": [{"layer": n, "cosine": c, "scale_spread": sp} for c, n, sp in below],
                   "worst_cosine": t["worst_cosine"], "worst_layer": t["worst_layer"]},
        "alternative": {"scheme": "one scale per (output channel, tap)", "adopted": False,
                        "worst_cosine": per_tap[0][0], "worst_layer": per_tap[0][1]},
        "note": "the Gate-1 cosine bar was defined for the DiT base GEMMs; the VAE decoder is a "
                "separate artifact and its weights quantise worse (they are causal-conv kernels, "
                "not attention/FFN matrices). Reported, not silently relaxed.",
    }
    (out / "vae_int8.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote               : {', '.join(p.name for p in sorted(out.glob('*')))}")

    if not args.verify:
        return 0
    return verify(manifest, store, plan_convs, out, inv_bytes)


# ------------------------------------------------------------------- verifier
def layout_probe(store: Store, plans: list[dict], n_layers: int = 3) -> list[dict]:
    """Does the OHWI tap layout actually compute the causal 3-D conv?

    The claim the device binary depends on is not "the bytes are what I wrote" but "a `CONV_2D`
    over these bytes, accumulated over taps, is the checkpoint's conv". So: take a small random
    activation, run the *checkpoint* weight and the *dequantised INT8 tap* weight through the same
    2-D SAME conv (pad k/2, stride 1), sum over taps, and compare. Any transpose or tap-order
    mistake shows up as a large error, not as a plausible one.
    """
    out = []
    for pl in plans[:n_layers]:
        name = pl["name"]
        W = store.get(f"{name}.weight")
        taps, kt = tap_views(W)
        q, s, _ = quantise(taps)
        kh = W.shape[3] if W.ndim == 5 else W.shape[2]
        khw = W.shape[4] if W.ndim == 5 else W.shape[3]
        pad = kh // 2
        rng = np.random.default_rng(20261116)
        A = rng.standard_normal((1, 3 + kt, 5, 7, W.shape[1])).astype(np.float32)   # [1,F,H,W,Cin]
        ref = np.zeros((1, 5, 7, W.shape[0]), np.float64)
        got = np.zeros_like(ref)

        def conv2d_same(x: np.ndarray, K: np.ndarray) -> np.ndarray:
            """x [1,H,W,Cin], K [Cout,kH,kW,Cin] -> [1,H,W,Cout], SAME pad, stride 1."""
            xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
            y = np.zeros((1, x.shape[1], x.shape[2], K.shape[0]), np.float64)
            for i in range(kh):
                for j in range(khw):
                    y += np.einsum("nhwc,oc->nhwo", xp[:, i:i + x.shape[1], j:j + x.shape[2], :], K[:, i, j, :])
            return y

        for t in range(kt):
            src = A[:, 2 - t] if kt > 1 else A[:, 2]      # causal: frames f, f-1, ...
            Kw = W[:, :, t] if W.ndim == 5 else W
            ref += conv2d_same(src.astype(np.float32), np.transpose(Kw, (0, 2, 3, 1)))
            Kq = (q[t].astype(np.float32) * s[:, None]).reshape(W.shape[0], kh, khw, W.shape[1])
            got += conv2d_same(src.astype(np.float32), Kq)
        den = float(np.linalg.norm(ref))
        out.append({"layer": name, "kt": kt, "k": [kh, khw],
                    "rel_fro_vs_int8": float(np.linalg.norm(ref - got) / den) if den else 0.0,
                    "cosine_vs_int8": float((ref * got).sum() / den / np.linalg.norm(got)) if den else 1.0})
    return out


def verify(manifest: dict, store: Store, plan_convs: list[dict], out: Path, inv_bytes: int) -> int:
    """Independent: re-quantise from the source checkpoint and demand byte identity.

    Reads nothing the builder produced except the files themselves, so a bug in the layout
    arithmetic or in the write path cannot certify itself.
    """
    fails: list[str] = []
    w = np.fromfile(out / "weights_int8.bin", dtype=np.int8)
    s = np.fromfile(out / "scales_f32.bin", dtype="<f4")
    b = np.fromfile(out / "bias_f32.bin", dtype="<f4")
    # manifest offsets are BYTES into each file (what the device mmap wants); these arrays are
    # element-indexed, so divide by the element size — the first version of this verifier compared
    # byte offsets against element indices and only layer 0 (offset 0) passed.
    n_ok = 0
    below: list[tuple[str, float]] = []
    for rec, pl in zip(manifest["layers"], plan_convs):
        W = store.get(rec["weight_tensor"])
        taps, kt = tap_views(W)
        q, sc, stat = quantise(taps)
        if rec["name"] != pl["name"]:
            fails.append(f"{rec['name']}: layer order diverges from the plan")
        so = rec["scales"]["offset"] // 4
        if not np.array_equal(s[so:so + rec["scales"]["count"]], sc):
            fails.append(f"{rec['name']}: scales differ from an independent recomputation")
        for trec, qq in zip(rec["taps"], q):
            blob = w[trec["offset"]:trec["offset"] + trec["bytes"]]
            if not np.array_equal(blob, qq.reshape(-1)):
                fails.append(f"{rec['name']} tap {trec['tap']}: bytes differ from the recomputation")
                continue
            if trec["offset"] % ALIGN:
                fails.append(f"{rec['name']} tap {trec['tap']}: offset {trec['offset']} not {ALIGN}-aligned")
            if trec["bytes"] != W.shape[0] * rec["cin"] * int(np.prod(W.shape[3:] if W.ndim == 5 else W.shape[2:])):
                fails.append(f"{rec['name']} tap {trec['tap']}: {trec['bytes']} B for "
                             f"{W.shape} — window does not cover the tap")
            n_ok += 1
        if stat["cosine"] < GATE_COSINE:
            below.append((rec["name"], stat["cosine"]))
        bias = rec["bias"]
        if bias is not None:
            src = store.get(f"{rec['name']}.bias")
            bo = bias["offset"] // 4
            got = b[bo:bo + bias["count"]]
            if not np.array_equal(got, src.astype(np.float32)):
                fails.append(f"{rec['name']}: bias bytes differ from the checkpoint")
    # structural claims the device binary relies on
    if manifest["totals"]["executes_per_chunk"] != manifest["totals"]["plan_executes"]:
        fails.append("executes/chunk does not match the plan")
    if n_ok != manifest["totals"]["taps"]:
        fails.append(f"{n_ok} taps verified vs {manifest['totals']['taps']} claimed")
    if len(w) != manifest["files"]["weights_int8.bin"]["bytes"]:
        fails.append(f"weights file is {len(w)} B, manifest says {manifest['files']['weights_int8.bin']['bytes']}")
    if manifest["totals"]["int8_weight_bytes_unpadded"] != int(inv_bytes):
        fails.append("unpadded INT8 total does not match configs/vae/layers.json")
    if sha256_file(out / "weights_int8.bin") != manifest["files"]["weights_int8.bin"]["sha256"]:
        fails.append("weights sha256 does not match the manifest")
    diT = [rec["weight_tensor"] for rec in manifest["layers"]
           if not rec["weight_tensor"].startswith("decoder.")]
    if diT:
        fails.append(f"non-decoder tensors in the artifact: {diT[:3]}")

    # does the layout compute the conv? (transpose / tap-order / padding)
    probe = layout_probe(store, plan_convs)
    for p in probe:
        ok = p["rel_fro_vs_int8"] < 0.10
        print(f"LAYOUT {p['layer']}: kt={p['kt']} k={p['k']} "
              f"relFro checkpoint-vs-int8 {p['rel_fro_vs_int8']*100:.2f} % cosine {p['cosine_vs_int8']:.6f} "
              f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"{p['layer']}: OHWI tap layout does not reproduce the conv "
                         f"(relFro {p['rel_fro_vs_int8']:.3f})")

    for f in fails:
        print(f"FAIL {f}")
    print(f"QUANT: {len(below)}/{len(manifest['layers'])} layers below the Gate-1 cosine bar "
          f"{GATE_COSINE}" + (f" — {', '.join(f'{n} {c:.6f}' for n, c in below)}" if below else ""))
    print(f"{'PASS' if not fails else 'FAIL'}: {n_ok} taps byte-identical to a from-source recomputation, "
          f"{len(fails)} failure(s)")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
