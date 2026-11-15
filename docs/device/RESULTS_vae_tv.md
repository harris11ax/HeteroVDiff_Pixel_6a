# `T_V` — VAE decode on the Tensor Edge TPU (`CONV_2D` temporal taps)

*Run 2026-11-15 18:10–18:18. Artifacts: `docs/device/conv_sweep_20261115_181754.json`
(the table below), `docs/device/conv_sweep_20261115_181008.json` (the batched
attempts that failed to compile). Op inventory: `configs/vae_decode_ops.json`,
produced by `phase2_profile/inventory_vae.py`. Harness: `tools/npu/conv_vae_host.py`
+ `mode=conv` in `tools/android/npublock/npu_block.cpp`.*

## 1. What has to be decoded, exactly

`T_V` is `null` because NNAPI has **no `CONV_3D`** and the Wan decoder is a stack
of causal 3-D convs. Before choosing a decomposition, the decode was walked
layer-by-layer off the real checkpoint header (no torch, no modelling): causal
padding `(kT-1, 0)` keeps `F_out == F_in`, temporal upsample is `F' = 2F-1`
(3 → 5 → 9), spatial upsample is 2× nearest + 3×3 conv (34×60 → 68×120 → 136×240
→ 272×480). The walk lands on **9 output frames at 272×480**, which is Preset A —
i.e. the shape rules are right, not just plausible.

| | |
|---|---|
| ops | 37 (36 convs + 1 full 3-D mid-block attention) |
| conv MACs | **5117.3 GMAC** (99.4 % of the layer) |
| mid-block 3-D attention | 32.4 GMAC (6120 tokens, 384 ch) |
| **total** | **5149.7 GMAC = 10.30 TFLOP per chunk** |

The MAC budget is flat across the stack: every 3×3×3 resnet conv in
`up_blocks.2` (192 ch @ 136×240) and `up_blocks.3` (96 ch @ 272×480) is
**292.4 GMAC**, six of each. Those two classes are 68 % of all decode MACs, which
is where a kernel effort would have to go.

## 2. The decomposition that runs, and the one that does not

Branch measured: each `kT=3` causal conv becomes **`kT` shifted 2-D `CONV_2D` ops**
over the (h, w) plane of one frame, accumulated into the output frame. The unit of
measurement is one `CONV_2D` at the layer's real shape — the same construction as
the `T_N` per-GEMM table.

**Finding: the batch dimension is the compiler's limit, not the byte count.**
`google-edgetpu` reports `conv2d_delegated=1` for every shape tried, including
ones it then cannot compile — the support query is not a viability test (same
lesson as the K sweep).

| CONV_2D input | per-frame bytes | result |
|---|---|---|
| N=3, 34×60×384 | 0.78 MB | compiles 141.6 ms, runs |
| N=5, 68×120×192 | 1.4 MB | **compile fails after 31.2 s** |
| N=3, 68×120×192 | 1.4 MB | **compile fails after 33.1 s** |
| N=3, 136×240×192 | 6.3 MB | **compile fails after 31.2 s** |
| N=9, 136×240×192 | 6.3 MB | **compile fails after 35.7 s** |
| **N=1, 136×240×192** | 6.3 MB | **compiles 263 ms, runs** |

The 37.6 MB tensor that fails as `N=3` compiles in 0.26 s as `N=1`. So the
per-frame decomposition is the only viable path: **`T_V` on the TPU = F separate
frame executes per temporal tap.** (Cost: 9 executes per tap instead of 1; not
free — see §4.)

## 3. Measured per-class rates (hot start — see §5)

`layer_ms = ms × ktaps × sub_ops`, `class_ms = layer_ms × ops_in_class`.

| class | N used | ms/op | GOPS | layer ms | ops | class ms |
|---|---|---|---|---|---|---|
| conv3³ 192→192 @136×240 | 1 | 21.099 | 1026.5 | 569.7 | 6 | **3418.0** |
| conv3³ 96→96 @272×480 | 1 | 31.605 | 685.3 | 853.3 | 6 | **5120.0** |
| conv3³ 384→384 @68×120 | 1 | 11.757 | 1842.2 | 176.4 | 5 | 881.8 |
| conv3³ 384→384 @34×60 | 3 | 8.874 | 1830.5 | 26.6 | 10 | 266.2 |
| conv3×3 384→192 @136×240 | 1 | 33.528 | 1292.0 | 301.8 | 1 | 301.8 |
| conv3×3 192→96 @272×480 | 1 | 53.874 | 804.0 | 484.9 | 1 | 484.9 |
| conv3³ 192→384 @68×120 | 1 | 7.239 | 1495.9 | 108.6 | 1 | 108.6 |
| time_conv 384→768 @68×120 | 1 | 12.437 | 387.0 | 186.6 | 1 | 186.6 |
| conv3×3 384→192 @68×120 | 1 | 7.707 | 1405.1 | 38.5 | 1 | 38.5 |
| conv3³ 96→3 @272×480 | 1 | 17.030 | 39.7 | 459.8 | 1 | 459.8 |
| time_conv 384→768 @34×60 | 3 | 14.805 | 243.8 | 44.4 | 1 | 44.4 |
| conv1×1 192→384 @68×120 | 5 | 89.284 | 67.4 | 89.3 | 1 | 89.3 |
| conv3³ 16→384 @34×60 | 3 | 4.094 | 165.3 | 12.3 | 1 | 12.3 |

**`T_V` (TPU, `CONV_2D` taps) = 11.412 s per chunk, 896.8 GOPS aggregate, over
100 % of the conv MACs.**

That is a *complete measured sum* over the conv work and a **lower bound on the
full decode**, because the mid-block 3-D attention (32.4 GMAC, 6120 tokens) has
no measured rate yet — it is not a conv and does not fit this harness.

Two rates in the table are suspicious and are kept as measured rather than
smoothed: `96→3 @272×480` (a 3-channel output, 39.7 GOPS) and `1×1 192→384`
(67.4 GOPS, 89 ms for 3.0 GMAC — 3× worse than the 96-channel 1×1-equivalent
shapes). Both are small-MAC ops dominated by fixed per-execute cost; neither can
move `T_V` by more than 0.5 s.

## 4. What this says about the NPU stage

- **`CONV_2D` is 3–6× more efficient than `FULLY_CONNECTED` on this HAL**: the
  best conv class here is 1842 GOPS against the 285 GOPS the FC path reached.
  The Edge TPU is a conv engine; the DiT block looks like 10 GEMMs to it and is
  being served by the wrong op class. That is a Phase-3 lever worth naming: re-
  expressing `ffn.up`/`attn` as `CONV_2D` with a 1×1 kernel would test whether the
  4.18× `T_N/T_G` penalty is the *op class* rather than the silicon.
- The 9 executes/tap cost is real: `96→96 @272×480` at N=1 is 31.6 ms, i.e. an
  input of 25 MB re-read per execute. Batching would fix that and does not
  compile.
- Decomposition is exact (frames are independent for a 2-D tap) except for
  spatially split fallbacks, which are marked `seam=True` in the artifact and
  were not needed here.

## 5. Caveat that must survive into the paper

**Every sample above is a HOT start and is not table-grade.** Zones during the
sweep: TPU 45 → 60 °C, `VIRTUAL-SKIN` 37.3 → 39.0 °C (its first throttle
threshold is 39.0 °C), battery 29.6 → 30.8 °C, so the gate
`max(TPU, skin, VIRTUAL-SKIN) ≤ battery + 3` was **never** satisfied (~32.6 °C
allowed vs 45 °C+ actual). The device was still warm from the failed batched
compiles and never cooled.

Direction of the error: throttling can only make these numbers *slower*, so
11.412 s is an **upper bound** on the cool-start TPU `T_V`. A gate-compliant
re-run of the two dominant classes (68 % of MACs) is the next device action:
`python tools/npu/cool_rerun.py`, then `conv_vae_host.py --run --only "c96->96"`.

## 6. Still TBD

| quantity | state |
|---|---|
| `T_V` TPU, conv MACs | **11.412 s/chunk** — complete measured sum, HOT start |
| `T_V` TPU, mid-block attention (32.4 GMAC) | TBD — no measured rate (not a conv) |
| `T_V` Mali (OpenCL) branch | TBD — no conv kernel written yet; the item needs a `conv3d`-as-im2col kernel over `gemm8.cl`, or the paper's 60×40 spatial tiling |
| co-run `T_G` | TBD |
| `k_b*` | TBD (blocked on `T_V` complete + co-run `T_G`) |
