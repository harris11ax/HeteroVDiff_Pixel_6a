# `T_V` — VAE decode: Tensor Edge TPU (`CONV_2D` taps) and Mali (`conv8_taps`)

*Runs 2026-11-15: TPU sweep 18:25:12 (gate-compliant, TPU 32.0 °C start) and 18:17:54 (hot,
kept as the throttling control); **Mali sweep 19:09:25** (TPU 26.0 °C / battery 23.1 °C at
start, `gemm_mali_20261115_190925.{json,md}`). Artifacts:
`docs/device/conv_sweep_20261115_182512.json`, `conv_sweep_20261115_181754.json`,
`conv_sweep_20261115_181008.json` (the batched attempts that failed to compile),
`gemm_mali_20261115_190925.*` (Mali). A 18:51:32 Mali attempt is **not** kept and must not be cited — the kernel
faulted and its artifact was a table of enqueue overheads (§6).
Op inventory: `configs/vae_decode_ops.json` from `phase2_profile/inventory_vae.py`.
Harnesses: `tools/npu/conv_vae_host.py` + `mode=conv` in `tools/android/npublock/npu_block.cpp`
(TPU); `tools/gemm/gemm_host.py --mode conv` + `conv8_taps` in
`tools/android/gemm/kernels/gemm8.cl` (Mali).*

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

## 5. The Mali branch — same classes, `conv8_taps` (implicit GEMM)

`tools/android/gemm/kernels/gemm8.cl :: conv8_taps`. The GPU has no `CONV_3D` problem to
solve: all `kT` temporal taps and `3x3` spatial taps are accumulated in **one dispatch** per
class, as an implicit GEMM over a zero-padded **NHWC** int8 input with the **channel fastest**
in K (`K = taps·s²·Cin`). That layout is what keeps the `char4` + `arm_dot_acc` inner loop of
the tuned GEMM: 4 consecutive K values are 4 consecutive channels of the same pixel. Cin is
padded to a multiple of 32 channels with zero channels *and* zero weights (`CinP` column
below), so a vector load never straddles a pixel and the padding contributes nothing.

Protocol: `cool_rerun.py --gate-slack 1.0 --confirm-samples 3`, start TPU **26.0 °C**, battery
23.1 → gate 26.1 (27.1 with the 1 °C slack, see §7), skin 24.4/25.2, `VIRTUAL-SKIN` 24.3;
TPU reached 34.0 °C by the end of the sweep. Variant `tiled_64x64x32_t8x4`. Heaviest classes
first, so the MACs that dominate run at the floor.

| class | N | ms/dispatch | GOPS | layer ms | ops | class ms |
|---|---|---|---|---|---|---|
| conv3³ 96→96 @272×480 | 1 | 85.840 | 378.5 | 772.6 | 6 | **4635.3** |
| conv3³ 192→192 @136×240 | 1 | 66.865 | 485.9 | 601.8 | 6 | **3610.7** |
| conv3³ 384→384 @68×120 | 1 | 68.167 | 476.6 | 340.8 | 5 | **1704.2** |
| conv3³ 384→384 @34×60 | 1 | 19.084 | 425.6 | 57.3 | 10 | **572.5** |
| conv3×3 192→96 @272×480 | 1 | 57.634 | 375.8 | 518.7 | 1 | 518.7 |
| conv3×3 384→192 @136×240 | 1 | 51.318 | 422.0 | 461.9 | 1 | 461.9 |
| conv3³ 192→384 @68×120 | 1 | 34.929 | 465.1 | 174.6 | 1 | 174.6 |
| time_conv 384→768 @68×120 | 1 | 17.357 | 415.9 | 86.8 | 1 | 86.8 |
| conv3×3 384→192 @68×120 | 1 | 12.949 | 418.1 | 64.7 | 1 | 64.7 |
| conv3³ 96→3 @272×480 | 1 | 42.783 | 23.7 | 385.0 | 1 | 385.0 |
| time_conv 384→768 @34×60 | 1 | 4.813 | 375.0 | 14.4 | 1 | 14.4 |
| conv1×1 192→384 @68×120 | 1 | 3.569 | 168.5 | 17.8 | 1 | 17.8 |
| conv3³ 16→384 @34×60 | 1 | 2.957 | 114.5 | 8.9 | 1 | 8.9 |

**`T_V` (Mali, `conv8_taps`) = 12.256 s per chunk, 417.5 GOPS aggregate, over 100 % of the
conv MACs.** Verified, not assumed: 7 of the 13 classes were run against `conv8_ref`, the
scalar int32 reference over the same buffers, and all 7 are **exact (0 mismatches, non-zero
output)** — the two 3³ classes, the temporal-1×1 class, the 1×1 class, the tiny-`Cout`=3 tail
and the `CinP`-padded `Cin=16` class. The six unverified classes are the ones whose scalar
reference would cost > 10 GMAC on one work-item per output; they use the same code path.

## 6. What the two branches say — the ordering flips here

| | TPU (`CONV_2D` taps) | Mali (`conv8_taps`) | faster |
|---|---|---|---|
| `T_V` | **10.554 s/chunk** (969.7 GOPS) | **12.256 s/chunk** (417.5 GOPS) | **TPU, 1.16×** |
| DiT block, same phone | 0.6758 s/block (210 GOPS) | 0.1618 s/block (877 GOPS) | Mali, 4.18× |

The DiT block's GEMMs are 4.18× faster on the GPU; the **decode's convs are 1.16× faster on
the NPU**. `CONV_2D` is the Edge TPU's native op (685–2024 GOPS measured across shapes) and
`FULLY_CONNECTED` is not, while on Mali the GEMM kernel is the tuned one (877 GOPS) and this
conv kernel reaches only 378–486 GOPS on the big classes — **48 % of the GEMM rate on the
same silicon**, i.e. the Mali conv path is the unoptimised side of the comparison.

Why the Mali conv is at 48 %: every work-group re-stages its 64-pixel activation tile once per
(tap, channel-block) — 27 taps × 3 channel blocks for the `96→96 @272×480` class — with no
spatial halo reuse, so the A-side traffic is 9× what a halo-staged tile would need (measured
677 MB per dispatch ⇒ 7.9 GB/s at 85.8 ms, so it is *not* DRAM-bound: the cost is the LDS
staging + barrier count per group). A shared halo tile per work-group is the obvious lever
and the GEMM's 877 GOPS is the ceiling it would approach; **no number for that variant is
claimed here** — it is the next measurement, not a model.

Consequence for the cut, which is the opposite of the block result: the decode term is the one
place the NPU should own outright, and `T_V > 65` GPU-blocks of work means the cut's suffix
assignment now depends on which engine holds the decode. Nothing in this file changes a locked
decision — it reports that `T_V` is TPU-favourable while `T_N` is not.

### The fault that cost one run (kept as evidence)

The 18:51:32 sweep was gate-compliant and produced `90185 GOPS` — impossible on a 1.77 TOPS
GPU. (Its artifact was deleted rather than committed: a table of enqueue overheads with a
GOPS column is worse than no table. Re-running the sweep is what recovered the number.) Every blocking read returned `CL_EXEC_STATUS_ERROR(-14)`: `Abase` had the 1-pixel border
added *and* the tap offset, so the last tap row ran ~46 KB past the padded frame buffer and
every dispatch faulted; the "times" were enqueue overhead only. Fixed (unpadded origin, the
border is the tap offset), and the harness now **refuses** a class whose warm-up or timed read
faults, and reports `UNVERIFIED` instead of `MATCH` when both buffers are zero — "0
mismatches between two zero buffers" is not a pass.

## 7. Still TBD

| quantity | state |
|---|---|
| `T_V` TPU, conv MACs | **10.554 s/chunk**, 969.7 GOPS, 100 % of conv MACs, gate-compliant start |
| `T_V` Mali, conv MACs | **12.256 s/chunk**, 417.5 GOPS, 100 % of conv MACs, 7/13 classes exact vs `conv8_ref` |
| `T_V` mid-block attention (32.4 GMAC) | TBD — no measured rate on either engine (not a conv); `T_V` stays a **lower bound** on both branches |
| Mali conv rate vs its own GEMM | 378–486 GOPS vs 877 GOPS — halo-staged activation tile is the next measurement (no number claimed) |
| co-run `T_G` | TBD |
| `k_b*` | TBD (blocked on the mid-block attention + co-run `T_G`) |

### Thermal-gate note (why `--gate-slack 1.0`)

The locked gate is `max(TPU, skin) <= battery + 3`. The TPU zone on this phone reports in
**whole degrees** (26.0 / 27.0) while the battery resolves 0.1 °C, so once the battery cools to
ambient — 23.1 °C here after ~25 min of cooling — the gate lands *between* two sensor steps
(26.1) and flickers PASS/FAIL on a device that has not changed temperature: the 19:06 attempt
passed on one sample (TPU 26.0 vs 26.1) and aborted on the confirm sample (TPU 27.0 vs 26.2).
`cool_rerun.py` therefore gained `--gate-slack` (default 0, logged) and `--confirm-samples`;
the Mali sweep started at TPU 26.0 °C with a 0-slack gate of 26.1 °C, i.e. it passes even with
the slack removed. This is reported as **at the idle floor**, and the 34.0 °C end-of-run zone
shows how much headroom the sweep actually used.
