#!/usr/bin/env python3
"""Per-tensor vs per-output-column INT8 weight scales: the parity cost, measured.

The locked quantisation is per-output-column symmetric INT8, but NNAPI's
FULLY_CONNECTED rejects TENSOR_QUANT8_SYMM_PER_CHANNEL at addOperation
(docs/device/RESULTS_npu_tn.md, finding #2). So the NPU stage can only express
per-tensor scales, and the fork is: accept the parity cost on the NPU stage, or
restrict the NPU to shapes with an equivalent path. This script measures the
cost instead of guessing it.

It is a PARITY measurement, not a latency one: it reads the real FP32 weights
out of the checkpoint and compares, per GEMM, what the two scale layouts give.
No device number is produced or modelled here, and nothing was quantised to
disk (the single stored INT8 copy rule is untouched).

    python phase1_quant/scale_parity.py --json docs/device/scale_parity.json

Reference arithmetic: W and the dequantised weights are evaluated in float64,
so the reported error is the quantiser's, not the emulator's.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO / "weights_raw" / "DCM_WAN" / "transformer" / "diffusion_pytorch_model.safetensors"

# The block's GEMM shapes (Preset A, block 0). Every one of the 30 blocks is
# shape-identical; --blocks samples extra blocks to confirm the error is stable.
BLOCK_GEMMS = [
    "ffn.net.0.proj.weight",      # [8960,1536]  up
    "ffn.net.2.weight",           # [1536,8960]  down
    "attn1.to_q.base_layer.weight",
    "attn1.to_k.base_layer.weight",
    "attn1.to_v.base_layer.weight",
    "attn1.to_out.0.base_layer.weight",
    "attn2.to_q.base_layer.weight",
    "attn2.to_k.base_layer.weight",
    "attn2.to_v.base_layer.weight",
    "attn2.to_out.0.base_layer.weight",
]


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return {k: v for k, v in hdr.items() if k != "__metadata__"}, 8 + n


class Store:
    """Read-only mmap of the safetensors payload; no tensor is copied unless used."""

    def __init__(self, path: Path):
        self.hdr, self.data_start = read_header(path)
        self.path = path
        self.arr = np.memmap(path, dtype=np.uint8, mode="r")

    def get(self, name: str) -> np.ndarray:
        m = self.hdr[name]
        if m["dtype"] != "F32":
            raise SystemExit(f"{name}: dtype {m['dtype']} not F32")
        a, b = m["data_offsets"]
        flat = np.frombuffer(self.arr, dtype="<f4", count=(b - a) // 4, offset=self.data_start + a)
        return flat.reshape(m["shape"]).astype(np.float64)


def quant_per_tensor(W: np.ndarray) -> tuple[np.ndarray, float]:
    s = float(np.abs(W).max()) / 127.0
    s = s if s > 0 else 1.0
    q = np.clip(np.rint(W / s), -127, 127)
    return q, s


def quant_per_column(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One FP32 scale per output column (row of the row-major [N_out, K] weight)."""
    s = np.abs(W).max(axis=1) / 127.0
    s[s == 0] = 1.0
    q = np.clip(np.rint(W / s[:, None]), -127, 127)
    return q, s


def metrics(ref: np.ndarray, test: np.ndarray) -> dict:
    d = test - ref
    nref = float(np.linalg.norm(ref))
    return {
        "rel_fro": float(np.linalg.norm(d) / nref) if nref else 0.0,
        "cosine": float((ref * test).sum() / (nref * np.linalg.norm(test))) if nref else 1.0,
        "max_abs": float(np.abs(d).max()),
    }


def gemm_case(store: Store, name: str, tokens: int, seed: int) -> dict:
    W = store.get(name)                      # [N_out, K]
    n_out, k = W.shape

    q_t, s_t = quant_per_tensor(W)
    q_c, s_c = quant_per_column(W)
    Wt = q_t * s_t                           # per-tensor dequant
    Wc = q_c * s_c[:, None]                  # per-column dequant

    # Activation: deterministic, INT8-quantised per tensor exactly as the device
    # FC path does it (one scale for the whole activation tile).
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((tokens, k))
    sa = float(np.abs(X).max()) / 127.0
    Xq = np.clip(np.rint(X / sa), -127, 127) * sa

    Y_ref = X @ W.T
    Y_t = Xq @ Wt.T
    Y_c = Xq @ Wc.T
    Y_w_only = X @ W.T                          # act-FP32 reference for weight-only error

    out = {
        "tensor": name,
        "shape": [n_out, k],
        "macs": n_out * k * tokens,
        "weight_only": {
            "per_tensor": metrics(W, Wt),
            "per_column": metrics(W, Wc),
        },
        "gemm_act_int8_per_tensor": {
            "per_tensor": metrics(Y_ref, Y_t),
            "per_column": metrics(Y_ref, Y_c),
        },
        "scales": {
            "per_tensor_scale": s_t,
            "per_column_min": float(s_c.min()),
            "per_column_max": float(s_c.max()),
            "per_column_spread": float(s_c.max() / s_c.min()),
            # how much of the per-tensor range one column actually needs:
            "worst_column_scale_ratio": float(s_c.min() / s_t),
        },
    }
    del W, q_c, q_t, Wt, Wc, X, Xq, Y_ref, Y_t, Y_c, Y_w_only
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--blocks", default="0", help="comma list of block indices to sample")
    ap.add_argument("--tokens", type=int, default=256, help="activation tokens per GEMM")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    store = Store(Path(args.ckpt))
    blocks = [int(b) for b in args.blocks.split(",")]
    t0 = time.time()

    cases = []
    for b in blocks:
        for suffix in BLOCK_GEMMS:
            name = f"blocks.{b}.{suffix}"
            if name not in store.hdr:
                print(f"  skip (absent): {name}")
                continue
            c = gemm_case(store, name, args.tokens, seed=1234 + b)
            cases.append(c)
            wt = c["weight_only"]
            g = c["gemm_act_int8_per_tensor"]
            print(f"{name:52s} {c['shape']}  "
                  f"W relF {wt['per_tensor']['rel_fro']*100:5.2f}%/{wt['per_column']['rel_fro']*100:5.2f}%  "
                  f"cos {wt['per_tensor']['cosine']:.6f}/{wt['per_column']['cosine']:.6f}  "
                  f"Y relF {g['per_tensor']['rel_fro']*100:5.2f}%/{g['per_column']['rel_fro']*100:5.2f}%  "
                  f"col-spread {c['scales']['per_column_spread']:.1f}x")

    # MAC-weighted aggregate over the sampled instances (per-MAC weighting, so a
    # 1536x1536 attention GEMM does not count the same as the 1536x8960 FFN).
    def wavg(key: str, layout: str) -> float:
        num = sum(c["macs"] * c["gemm_act_int8_per_tensor"][layout][key] for c in cases)
        den = sum(c["macs"] for c in cases)
        return num / den

    agg = {
        "n_gemms": len(cases),
        "blocks": blocks,
        "tokens": args.tokens,
        "mac_weighted_gemm_rel_fro": {
            "per_tensor": wavg("rel_fro", "per_tensor"),
            "per_column": wavg("rel_fro", "per_column"),
        },
        "mac_weighted_gemm_cosine": {
            "per_tensor": wavg("cosine", "per_tensor"),
            "per_column": wavg("cosine", "per_column"),
        },
        "worst_gemm_rel_fro": {
            "per_tensor": max(c["gemm_act_int8_per_tensor"]["per_tensor"]["rel_fro"] for c in cases),
            "per_column": max(c["gemm_act_int8_per_tensor"]["per_column"]["rel_fro"] for c in cases),
        },
    }
    print()
    print(f"MAC-weighted GEMM output rel. Frobenius error: per-tensor "
          f"{agg['mac_weighted_gemm_rel_fro']['per_tensor']*100:.2f}%  vs  per-column "
          f"{agg['mac_weighted_gemm_rel_fro']['per_column']*100:.2f}%")
    print(f"MAC-weighted cosine: per-tensor "
          f"{agg['mac_weighted_gemm_cosine']['per_tensor']:.6f}  vs  per-column "
          f"{agg['mac_weighted_gemm_cosine']['per_column']:.6f}")
    print(f"worst single GEMM rel. error: per-tensor "
          f"{agg['worst_gemm_rel_fro']['per_tensor']*100:.2f}%  vs  per-column "
          f"{agg['worst_gemm_rel_fro']['per_column']*100:.2f}%")
    print(f"({time.time()-t0:.1f}s, host-side parity on the real FP32 weights; no device rate claimed)")

    if args.json:
        out = {
            "kind": "scale_parity_host",
            "checkpoint": str(args.ckpt),
            "note": ("host-side exact-integer parity on the real FP32 weights; float64 reference. "
                     "Not a device latency measurement."),
            "activation": f"seeded standard normal, {args.tokens} tokens, per-tensor INT8 activation scale",
            "aggregate": agg,
            "cases": cases,
            "elapsed_s": time.time() - t0,
        }
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
