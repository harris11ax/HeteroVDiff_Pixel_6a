#!/usr/bin/env python3
"""Phase 1 deliverable 3: flatten the Wan `AutoencoderKLWan` decoder into an executable plan.

`configs/vae_decode_ops.json` (phase2_profile/inventory_vae.py) is the MAC inventory the `T_V`
measurement was taken against. This script turns the same walk into the two **execution plans**
the executors (Phase 3) consume, plus the scaling and the tile schedule:

  configs/vae/layers.json      ordered layer list: op kind, weight tensor + shape, stride, causal
                               temporal padding, frames of input to cache across a chunk boundary,
                               in/out (f,h,w) at Preset A, MACs
  configs/vae/plan_npu.json  NNAPI plan: every `kT=3` causal conv -> `kT` shifted 2-D `CONV_2D`
                               taps + accumulate, one execute per frame (batch will not compile),
                               with the measured per-class ms bound in from the gate-compliant sweep
  configs/vae/plan_mali.json OpenCL plan: the whole 3-D conv as ONE `conv8_taps` dispatch, tile
                               shape from the deployed kernel, weight-row repack spec, measured ms
  configs/vae/scaling.json   `latents_mean` / `latents_std` (z must be de-normalised first)
  configs/vae/tiles.json     tile schedule at the latent plane + per-level halo

No torch: the walk reads the safetensors header only, and the shape rules are the same ones
`inventory_vae.py` validated (causal pad `(kT-1, 0)` -> `F_out == F_in`; temporal upsample
`F' = 2F-1`; spatial upsample 2x nearest + 3x3 conv) — which is why the export can be *checked*
against the measured inventory instead of trusted.

Every rate quoted here is a measured number with its source file named. Nothing is modelled; a
layer with no measured class keeps `measured` = null.

    python phase1_quant/vae_export.py
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_VAE = REPO / "weights_raw" / "DCM_WAN" / "vae" / "diffusion_pytorch_model.safetensors"
VAE_CFG = REPO / "weights_raw" / "DCM_WAN" / "vae" / "config.json"
OPS_JSON = REPO / "configs" / "vae_decode_ops.json"
NPU_SWEEP = REPO / "docs" / "device" / "conv_sweep_20261115_182512.json"      # gate-compliant (TPU 32.0 C)
MALI_SWEEP = REPO / "docs" / "device" / "gemm_mali_20261115_190925.json"      # Mali conv8_taps
OUT = REPO / "configs" / "vae"   # small, deterministic, reviewable in a diff (like configs/vae_decode_ops.json)

# Preset A (configs/design_locks.json -> workload.presets.preset_a)
P = {"frames": 9, "H": 272, "W": 480, "F_z": 3, "H_z": 34, "W_z": 60}
# conv8_taps deployment (tools/android/gemm/kernels/gemm8.cl): TM 8, TN 8, BM 64, BN 64, BK 32,
# char4 operands => KV = 8 char4 per BK slice, and Cin padded to a multiple of 4*KV = 16.
MALI_TILE = {"TM": 8, "TN": 8, "BM": 64, "BN": 64, "BK": 32, "cin_align": 16,
             "global": "ceil(Cout/BN) x ceil(H*W/BM) x (BM/TM)*(BN/TN) threads"}
# NNAPI CONV_2D: no CONV_3D on this HAL (Gate 0), and a batch dim > 1 fails to compile above
# ~1 MB/frame (RESULTS_vae_tv.md section 2), so the plan is one execute per (frame, tap).
NPU_BATCH_N = 1


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return {k: v for k, v in json.loads(f.read(n)).items() if k != "__metadata__"}


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def main() -> int:
    hdr = read_header(DEFAULT_VAE)
    cfg = json.loads(VAE_CFG.read_text(encoding="utf-8"))
    inv = json.loads(OPS_JSON.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- measured per-class rates, bound by geometry ---------------------
    npu_rate: dict[tuple, dict] = {}
    if NPU_SWEEP.exists():
        sw = json.loads(NPU_SWEEP.read_text(encoding="utf-8"))
        best: dict[str, dict] = {}
        for a in sw.get("attempts", []):
            if not a.get("ok"):
                continue
            cls = a.get("class")
            if cls not in best or (a.get("ms") or 1e9) < (best[cls].get("ms") or 1e9):
                best[cls] = a
        for cname, c in sw.get("classes", {}).items():
            a = best.get(cname)
            if not a:
                continue
            npu_rate[(c["cin"], c["cout"], c["k"], c["h"], c["w"])] = {
                "class": cname, "ms_per_conv2d": a["ms"], "gops": a.get("gops"),
                "ktaps": c["ktaps"], "ops_in_class": c["ops"], "sub_ops": a.get("sub_ops", 1),
                "N": a.get("N"), "source": NPU_SWEEP.name,
            }

    mali_rate: dict[tuple, dict] = {}
    if MALI_SWEEP.exists():
        ms = json.loads(MALI_SWEEP.read_text(encoding="utf-8")).get("parsed", {})
        for c in ms.get("conv", []):
            mali_rate[(c["Cin"], c["Cout"], c["s"], c["H"], c["W"])] = {
                "class": c["class"], "ms": c["ms"], "gops": c.get("gops"),
                "layer_ms": c.get("layer_ms"), "cls_ms": c.get("cls_ms"),
                "ops_in_class": c.get("ops"), "source": MALI_SWEEP.name,
            }

    # ---- walk the decoder in module order --------------------------------
    layers: list[dict] = []
    F, H, W = P["F_z"], P["H_z"], P["W_z"]

    def emit(kind, name, weight, f, h, w, upsample=None):
        """One decoder op at output grid (f,h,w) of the layer's own resolution."""
        if weight not in hdr:                      # e.g. a resnet with no conv_shortcut tensor
            return
        shp = hdr[weight]["shape"]
        rec = {"idx": len(layers), "name": name, "kind": kind, "weight": weight,
               "weight_shape": shp, "out_fhw": [f, h, w]}
        if kind == "attention":
            nt = f * h * w
            rec.update({"tokens": nt, "channels": 384,
                        "macs": 1152 * 384 * nt + 2 * nt * nt * 384 + 384 * 384 * nt,
                        "temporal_cache_frames": 0})
            layers.append(rec)
            return
        co, ci, kt = shp[0], shp[1], (shp[2] if len(shp) == 5 else 1)
        kh, kw = (shp[3], shp[4]) if len(shp) == 5 else (shp[2], shp[3])
        stride = 1
        rec.update({
            "weight_bytes_if_int8": co * ci * kt * kh * kw,
            "cin": ci, "cout": co, "k": [kt, kh, kw], "stride": stride,
            "causal_pad_T": [kt - 1, 0],
            "causal_note": "taps read frames f, f-1, ... f-(kT-1); F_out == F_in at every layer",
            "temporal_cache_frames": kt - 1,
            "macs": co * ci * kt * kh * kw * f * h * w,
        })
        if upsample:
            rec["upsample"] = upsample
        layers.append(rec)

    # conv_in, mid attention, mid resnets
    emit("conv3d", "decoder.conv_in", "decoder.conv_in.weight", F, H, W)
    emit("attention", "decoder.mid_block.attentions.0", "decoder.mid_block.attentions.0.to_qkv.weight", F, H, W)
    for r in range(2):
        for cv in ("conv1", "conv2", "conv_shortcut"):
            nm = f"decoder.mid_block.resnets.{r}"
            emit("conv3d", f"{nm}.{cv}", f"{nm}.{cv}.weight", F, H, W)

    # up_blocks: reversed dim_mult = 4, 4, 2, 1; 3 resnets per level; 2 upsamples per level
    for ub in range(4):
        for r in range(3):
            for cv in ("conv1", "conv2", "conv_shortcut"):
                nm = f"decoder.up_blocks.{ub}.resnets.{r}"
                emit("conv3d", f"{nm}.{cv}", f"{nm}.{cv}.weight", F, H, W)
        up = f"decoder.up_blocks.{ub}.upsamplers.0"
        if f"{up}.time_conv.weight" in hdr:
            emit("conv3d", f"{up}.time_conv", f"{up}.time_conv.weight", F, H, W,
                 upsample={"axis": "T", "rule": "F' = 2F-1", "kind": "3-tap causal time_conv then frame interleave"})
            F = 2 * F - 1
        if f"{up}.resample.1.weight" in hdr:
            H, W = 2 * H, 2 * W
            emit("conv3d", f"{up}.resample.1", f"{up}.resample.1.weight", F, H, W,
                 upsample={"axis": "HW", "rule": "2x nearest then 3x3 conv", "kind": "nearest+conv"})

    emit("conv3d", "decoder.conv_out", "decoder.conv_out.weight", F, H, W)

    # ---- check the walk against the measured inventory -------------------
    inv_names = {o["name"]: o["macs"] for o in inv["ops"]}
    mine = {l["name"]: l["macs"] for l in layers}
    if set(inv_names) != set(mine):
        raise SystemExit(f"walk mismatch vs configs/vae_decode_ops.json: "
                         f"{len(set(inv_names) - set(mine))} missing {sorted(set(inv_names) - set(mine))[:4]}, "
                         f"{len(set(mine) - set(inv_names))} extra {sorted(set(mine) - set(inv_names))[:4]}")
    bad = [n for n in mine if mine[n] != inv_names[n]]
    if bad:
        raise SystemExit(f"MAC mismatch on {len(bad)} layers, e.g. {bad[0]}: "
                         f"{mine[bad[0]]:,} vs inventory {inv_names[bad[0]]:,}")
    total_macs = sum(mine.values())

    # ---- bind measured rates layer by layer ------------------------------
    for l in layers:
        if l["kind"] == "attention":
            l["measured"] = None
            l["measured_note"] = ("no rate on either engine: 6120 tokens of 384-ch 3-D attention; "
                                  "0.63 % of decode MACs, so T_V stays a lower bound until it is measured")
            continue
        if l["kind"] == "upsample_spatial":
            l["measured"] = None
            l["measured_note"] = "nearest-neighbour resample; the 3x3 conv that follows is its own layer"
            continue
        f_, h_, w_ = l["out_fhw"]
        key_npu = (l["cin"], l["cout"], l["k"][1], h_, w_)     # sweep keys on the 2-D spatial kernel
        r = npu_rate.get(key_npu)
        l["measured"] = None
        if r:
            l["measured"] = {
                "tpu": {"class": r["class"], "ms_per_conv2d": r["ms_per_conv2d"],
                        "executes": r["ktaps"] * r["sub_ops"], "N": r["N"],
                        "layer_ms": r["ms_per_conv2d"] * r["ktaps"] * r["sub_ops"],
                        "source": r["source"]},
            }
        key_mali = (l["cin"], l["cout"], l["k"][1], h_, w_)
        rm = mali_rate.get(key_mali)
        if rm:
            l["measured"] = l["measured"] or {}
            l["measured"]["mali"] = {"class": rm["class"], "ms": rm["ms"],
                                     "dispatching": "one conv8_taps dispatch covering every tap",
                                     "layer_ms": rm["layer_ms"], "source": rm["source"]}
        if l["measured"] is None:
            l["measured_note"] = "no measured class for this geometry"

    # ---- plan_npu: tap decomposition -------------------------------------
    npu_layers = []
    npu_total_ms = 0.0
    for l in layers:
        if l["kind"] != "conv3d":
            continue
        f_, h_, w_ = l["out_fhw"]
        kt = l["k"][0]
        pads_h = "same" if l["k"][1] == 3 else "none"
        taps = []
        for t in range(kt):
            taps.append({
                "tap": t,
                "input_frame": f"f-{t}" if t else "f",
                "weight_slice": [0, t * l["cout"] * l["cin"] * l["k"][1] * l["k"][2],
                                 l["cout"] * l["cin"] * l["k"][1] * l["k"][2]],
                "kernel": [1, l["k"][1], l["k"][2]],
                "pad": [pads_h, "same" if l["k"][2] == 3 else "none"],
                "conv2d": [l["cout"], l["cin"], h_, w_],
            })
        ms = None
        if l["measured"] and "tpu" in l["measured"]:
            ms = l["measured"]["tpu"]["layer_ms"]
            npu_total_ms += ms
        npu_layers.append({
            "name": l["name"], "frames": f_, "taps": taps,
            "executes": kt * f_, "batch": NPU_BATCH_N,
            "accumulate": "int32 accumulate across the kT taps, then per-tensor dequant (epilogue)",
            "macs": l["macs"], "measured_ms_per_chunk": ms,
            "compile_note": "one graph per (layer, tap) — a graph holding several ops risks the "
                            "whole-block Compilation_finish reboot (RESULTS_npu_tn.md finding #1)",
        })
    npu_plan = {
        "kind": "vae_plan_npu", "engine": "Tensor Edge TPU via NNAPI",
        "preset_a_output_fhw": [P["frames"], P["H"], P["W"]],
        "latent_fhw": [P["F_z"], P["H_z"], P["W_z"]],
        "constraints": {
            "conv3d": "absent from NNAPI — every kT=3 causal conv becomes kT shifted CONV_2D taps",
            "batch": f"N={NPU_BATCH_N} only: N>1 fails to compile above ~1 MB/frame (measured)",
            "exactness": "exact under a 2-D tap; no seam",
        },
        "total_executes_per_chunk": sum(x["executes"] for x in npu_layers),
        "measured_total_ms": round(npu_total_ms, 1),
        "measured_source": f"{NPU_SWEEP.name} (gate-compliant, TPU 32.0 C start) via docs/device/RESULTS_vae_tv.md",
        "unmeasured": ["decoder.mid_block.attentions.0 (3-D attention, 32.4 GMAC)",
                       "spatial resample convolutions are conv8_taps/conv2d of their own class"],
        "layers": npu_layers,
    }

    # ---- plan_mali: direct 3-D conv --------------------------------------
    mali_layers = []
    mali_total_ms = 0.0
    for l in layers:
        if l["kind"] != "conv3d":
            continue
        f_, h_, w_ = l["out_fhw"]
        cinp = ceil_div(l["cin"], MALI_TILE["cin_align"]) * MALI_TILE["cin_align"]
        nst = l["k"][0] * l["k"][1] * l["k"][2]
        wg = ceil_div(l["cout"], MALI_TILE["BN"]) * ceil_div(h_ * w_, MALI_TILE["BM"])
        ms = None
        if l["measured"] and "mali" in l["measured"]:
            ms = l["measured"]["mali"]["layer_ms"]
            mali_total_ms += ms
        mali_layers.append({
            "name": l["name"], "frames": f_, "M": h_ * w_,
            "cin": l["cin"], "cin_padded": cinp, "cout": l["cout"],
            "k": l["k"], "s": l["k"][1], "taps": l["k"][0], "nst": nst,
            "k_total": nst * cinp, "weight_row_stride_char4": nst * cinp // 4,
            "workgroups": wg,
            "dispatch": "one conv8_taps dispatch per layer over all taps, per frame",
            "repack": f"[Cout, {nst * cinp}] row-major, k = ((t*s+ky)*s+kx)*CinP + c; "
                      f"Cin padded {l['cin']}->{cinp} with zero channels+weights",
            "kernel": "conv8_taps", "tile": MALI_TILE,
            "macs": l["macs"], "measured_ms_per_chunk": ms,
        })
    mali_plan = {
        "kind": "vae_plan_mali", "engine": "Mali-G78 MP20 via OpenCL",
        "deployed_kernel": {"file": "tools/android/gemm/kernels/gemm8.cl", "name": "conv8_taps",
                            "tile": MALI_TILE, "reference": "conv8_ref (exact int32, one work-item per output)"},
        "total_dispatches_per_chunk": sum(x["frames"] for x in mali_layers),
        "measured_total_ms": round(mali_total_ms, 1),
        "measured_source": f"{MALI_SWEEP.name} via docs/device/RESULTS_vae_tv.md",
        "one_copy_note": "the weight-row repack is a reshuffle of weights that are INT8 in the same "
                         "artifact — no second copy of the base weights is created",
        "layers": mali_layers,
    }

    # ---- scaling + tiles -------------------------------------------------
    scaling = {
        "kind": "vae_latent_scaling",
        "z_dim": cfg["z_dim"],
        "latents_mean": cfg["latents_mean"],
        "latents_std": cfg["latents_std"],
        "denorm": "z_phys = z * latents_std[c] + latents_mean[c]  (per channel, before decoder.conv_in)",
        "norm": "z = (z_phys - latents_mean[c]) / latents_std[c]  (encoder side, for calibration only)",
        "note": "16 channels; a mismatch here shows up as a global colour/tone shift, not as noise",
    }
    tile_px = 60 * 40
    latent_px = P["F_z"] * P["H_z"] * P["W_z"]
    tiles = {
        "kind": "vae_tile_schedule",
        "latent_fhw": [P["F_z"], P["H_z"], P["W_z"]],
        "latent_pixels": latent_px,
        "spec_tile_latent_pixels": tile_px,
        "spec_tile_note": f"docs/05_phase_plans.md asks for a 60x40-latent-pixel tile schedule; the "
                          f"Preset-A latent plane is {P['H_z']}x{P['W_z']} = {P['H_z']*P['W_z']} pixels "
                          f"< {tile_px}, so the schedule degenerates to ONE tile — no seams",
        "tiles": 1,
        "per_frame_executes": {"tpu": sum(x["frames"] * len(x["taps"]) for x in npu_layers),
                               "mali": sum(x["frames"] for x in mali_layers)},
        "halo": {"spatial_px": 1, "temporal_frames": 2,
                 "note": "kH=kW=3 needs a 1-pixel halo; kT=3 needs 2 previous input frames (causal)"},
        "levels": [{"name": "latent", "fhw": [P["F_z"], P["H_z"], P["W_z"]], "tiles": 1},
                   {"name": "after up_blocks.0", "fhw": [5, 68, 120], "tiles": 1},
                   {"name": "after up_blocks.1", "fhw": [9, 136, 240], "tiles": 1},
                   {"name": "after up_blocks.2", "fhw": [9, 272, 480], "tiles": 1}],
        "chunk_stream": "each chunk's decode is independent of the previous chunk's frames "
                        "(the DiT side carries the trajectory); the temporal cache is only "
                        "within-chunk, so no cross-chunk VAE state is needed",
    }

    # ---- totals + verdict ------------------------------------------------
    assert len(layers) == len(inv["ops"]), f"{len(layers)} layers vs {len(inv['ops'])} inventory ops"
    totals = {
        "layers": len(layers),
        "convs": sum(1 for l in layers if l["kind"] == "conv3d"),
        "attentions": sum(1 for l in layers if l["kind"] == "attention"),
        "total_macs": total_macs,
        "inventory_total_macs": inv["total_macs"],
        "macs_match": total_macs == inv["total_macs"],
        "npu_measured_total_ms": round(npu_total_ms, 1),
        "mali_measured_total_ms": round(mali_total_ms, 1),
        "npu_measured_total_s": round(npu_total_ms / 1000, 3),
        "mali_measured_total_s": round(mali_total_ms / 1000, 3),
        "reference_npu_s": 10.554,
        "reference_mali_s": 12.256,
        "output_fhw": [F, H, W],
        "output_ok": [F, H, W] == [P["frames"], P["H"], P["W"]],
    }

    (OUT / "layers.json").write_text(json.dumps({
        "kind": "vae_decoder_layers", "preset": "preset_a",
        "source_checkpoint": str(DEFAULT_VAE.relative_to(REPO)),
        "walk": "module order; causal pad (kT-1,0); F'=2F-1 on temporal upsample; 2x nearest + 3x3 on spatial",
        "totals": totals, "layers": layers,
    }, indent=2), encoding="utf-8")
    (OUT / "plan_npu.json").write_text(json.dumps(npu_plan, indent=2), encoding="utf-8")
    (OUT / "plan_mali.json").write_text(json.dumps(mali_plan, indent=2), encoding="utf-8")
    (OUT / "scaling.json").write_text(json.dumps(scaling, indent=2), encoding="utf-8")
    (OUT / "tiles.json").write_text(json.dumps(tiles, indent=2), encoding="utf-8")

    print(f"layers              : {totals['layers']} ({totals['convs']} convs + "
          f"{totals['attentions']} 3-D attention), output {totals['output_fhw']} "
          f"({'OK' if totals['output_ok'] else 'MISMATCH'})")
    print(f"MACs                : {total_macs/1e9:.2f} GMAC vs inventory {inv['total_macs']/1e9:.2f} GMAC "
          f"({'MATCH' if totals['macs_match'] else 'MISMATCH'})")
    print(f"measured, bound in  : TPU {totals['npu_measured_total_s']:.3f} s/chunk "
          f"(ref 10.554), Mali {totals['mali_measured_total_s']:.3f} s/chunk (ref 12.256)")
    print(f"executes            : TPU {npu_plan['total_executes_per_chunk']} CONV_2D executes/chunk "
          f"(N=1), Mali {mali_plan['total_dispatches_per_chunk']} dispatches/chunk")
    print(f"wrote               : {', '.join(p.name for p in sorted(OUT.glob('*.json')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
