# `T_V` — VAE decode on the Tensor Edge TPU (`CONV_2D` temporal taps)

*Run 2026-11-15 18:25:12 (gate-compliant, the table below) and 18:17:54 (hot, kept
as the throttling control). Artifacts: `docs/device/conv_sweep_20261115_182512.json`,
`docs/device/conv_sweep_20261115_181754.json`, `docs/device/conv_sweep_20261115_181008.json`
(the batched attempts that failed to compile). Op inventory: `configs/vae_decode_ops.json`
from `phase2_profile/inventory_vae.py`. Harness: `tools/npu/conv_vae_host.py` +
`mode=conv` in `tools/android/npublock/npu_block.cpp`.*

## 1. What has to be decoded, exactly

`T_V` was `null` because NNAPI has **no `CONV_3D`** and the Wan decoder is a stack of
causal 3-D convs. Before choosing a decomposition the decode was walked layer-by-layer
off the real checkpoint header (no torch, no modelling): causal padding `(kT-1, 0)`
keeps `F_out == F_in`, temporal upsample is `F' = 2F-1` (3 → 5 → 9), spatial upsample
is 2× nearest + 3×3 conv (34×60 → 68×120 → 136×240 → 272×480). The walk lands on
**9 output frames at 272×480** — Preset A — so the shape rules are validated, not
assumed.

| | |
|---|---|
| ops | 37 (36 convs + 1 full 3-D mid-block attention) |
| conv MACs | **5117.3 GMAC** (99.4 % of the layer) |
| mid-block 3-D attention | 32.4 GMAC (6120 tokens, 384 ch) |
| **total** | **5149.7 GMAC = 10.30 TFLOP per chunk** |

The MAC budget is flat across the stack: every 3×3×3 resnet conv in `up_blocks.2`
(192 ch @ 136×240) and `up_blocks.3` (96 ch @ 272×480) is **292.4 GMAC**, six of each —
those two classes are **68 % of all decode MACs**.

## 2. The decomposition that runs, and the one that does not

Branch measured: each `kT=3` causal conv becomes **`kT` shifted 2-D `CONV_2D` ops** over
the (h, w) plane of one frame, accumulated into the output frame. Unit of measurement =
one `CONV_2D` at the layer's real shape, the same construction as the `T_N` per-GEMM table.

**Finding: the batch dimension is the compiler's limit, not the byte count.** The
`google-edgetpu` support query reports `conv2d_delegated=1` for every shape below,
including the ones it then cannot compile — validation is again not a viability test.

| CONV_2D input | per-frame bytes | result |
|---|---|---|
| N=3, 34×60×384 | 0.78 MB | compiles 141.6 ms, runs |
| N=5, 68×120×192 | 1.4 MB | **compile fails after 31.2 s** |
| N=3, 68×120×192 | 1.4 MB | **compile fails after 33.1 s** |
| N=3, 136×240×192 | 6.3 MB | **compile fails after 31.2 s** |
| N=9, 136×240×192 | 6.3 MB | **compile fails after 35.7 s** |
| **N=1, 136×240×192** | 6.3 MB | **compiles 263 ms, runs** |

The 37.6 MB tensor that fails as `N=3` compiles in 0.26 s as `N=1`. So the per-frame
decomposition is the only viable path: **`T_V` on the TPU = F separate frame executes
per temporal tap.** It is exact (frames are independent under a 2-D tap); the spatial
split fallbacks in the ladder are marked `seam=True` and were not needed.

## 3. Measured per-class rates — GATE-COMPLIANT run

Sweep start 18:25:12: TPU **32.0 °C**, `skin_therm1` 30.5, `VIRTUAL-SKIN` 30.3, battery
29.1 → gate `max(TPU, skin, VIRTUAL-SKIN) ≤ battery + 3` = 32.1 °C, **PASS** (confirmed
on a second sample 10 s later). The two dominant classes ran first, at that floor; the
device reached 37 °C by the last class (all of which together are < 1 % of `T_V`).
Every class compiled and ran on the first attempt.

`layer_ms = ms × ktaps × sub_ops`, `class_ms = layer_ms × ops_in_class`.

| class | N used | ms/op | GOPS | layer ms | ops | class ms | hot ms (Δ) |
|---|---|---|---|---|---|---|---|
| conv3³ 192→192 @136×240 | 1 | 18.832 | 1150.1 | 508.5 | 6 | **3050.8** | 21.099 (+12.0 %) |
| conv3³ 96→96 @272×480 | 1 | 30.117 | 719.1 | 813.2 | 6 | **4879.0** | 31.605 (+4.9 %) |
| conv3³ 384→384 @68×120 | 1 | 10.701 | 2024.0 | 160.5 | 5 | 802.6 | 11.757 (+9.9 %) |
| conv3³ 384→384 @34×60 | 3 | 8.326 | 1951.0 | 25.0 | 10 | 249.8 | 8.874 (+6.6 %) |
| conv3×3 384→192 @136×240 | 1 | 31.927 | 1356.7 | 287.3 | 1 | 287.3 | 33.528 (+5.0 %) |
| conv3×3 192→96 @272×480 | 1 | 48.321 | 896.4 | 434.9 | 1 | 434.9 | 53.874 (+11.5 %) |
| conv3³ 192→384 @68×120 | 1 | 7.712 | 1404.2 | 115.7 | 1 | 115.7 | 7.239 (−6.1 %) |
| time_conv 384→768 @68×120 | 1 | 11.695 | 411.5 | 175.4 | 1 | 175.4 | 12.437 (+6.3 %) |
| conv3×3 384→192 @68×120 | 1 | 7.797 | 1388.9 | 39.0 | 1 | 39.0 | 7.707 (−1.2 %) |
| conv3³ 96→3 @272×480 | 1 | 15.997 | 42.3 | 431.9 | 1 | 431.9 | 17.030 (+6.5 %) |
| time_conv 384→768 @34×60 | 3 | 15.948 | 226.3 | 47.8 | 1 | 47.8 | 14.805 (−7.7 %) |
| conv1×1 192→384 @68×120 | 1 | 5.878 | 204.7 | 29.4 | 1 | 29.4 | 89.284 (**+1419 %**) |
| conv3³ 16→384 @34×60 | 3 | 3.607 | 187.6 | 10.8 | 1 | 10.8 | 4.094 (+13.5 %) |

**`T_V` (TPU, `CONV_2D` taps) = 10.554 s per chunk, 969.7 GOPS aggregate, over 100 %
of the conv MACs** — a complete measured sum, gate-compliant at the start.

Two caveats on the table, stated rather than smoothed:

- The `96→3 @272×480` (42.3 GOPS) and `1×1 192→384` (204.7 GOPS) rates are poor shapes
  dominated by fixed per-execute cost; together they are 4.4 % of `T_V`.
- The hot run's one absurd sample (89.3 ms for a 3.0 GMAC 1×1 conv, 15× the cool value)
  is what a throttled/contended sample looks like. It is the reason the cool re-run was
  taken rather than averaging both.

`T_V` here is a **lower bound on the full decode**: the mid-block 3-D attention
(32.4 GMAC, 6120 tokens) is not a conv and has no measured rate yet (0.6 % of MACs).

## 4. What this says about the NPU stage

- **`CONV_2D` is 2.5–7× more efficient than `FULLY_CONNECTED` on this HAL**: 719–2024
  GOPS here against 285 GOPS for the best FC shape. The Edge TPU is a conv engine and
  the DiT block's 10 GEMMs are being served by the wrong op class. That is a Phase-3
  lever: re-express `ffn.up`/`attn` as 1×1 `CONV_2D` and test whether the 4.18×
  `T_N/T_G` penalty is the op class rather than the silicon.
- The per-frame cost is real: `96→96 @272×480` at N=1 re-reads a 25 MB input per
  execute (30.1 ms); batching would fix that and does not compile.
- `T_V = 10.554 s` against `T_G = 0.1618 s/block` means the decode is worth **65 GPU
  blocks**. If it ran on the NPU the block-rate asymmetry shrinks to the same order as
  the decode — this is the term that will move `k_b*` most, so it needs to be right.

## 5. Still TBD

| quantity | state |
|---|---|
| `T_V` TPU, conv MACs | **10.554 s/chunk**, 969.7 GOPS, 100 % of conv MACs, gate-compliant start |
| `T_V` TPU, mid-block attention (32.4 GMAC) | TBD — no measured rate (not a conv) |
| `T_V` Mali (OpenCL) branch | TBD — no conv kernel yet; needs `conv3d`-as-im2col over `gemm8.cl` or the paper's 60×40 tiling |
| co-run `T_G` | TBD |
| `k_b*` | TBD (blocked on `T_V` complete + co-run `T_G`) |
