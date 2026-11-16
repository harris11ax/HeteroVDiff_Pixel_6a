# HANDOFF — HeteroVDiff on Pixel 6a

*Updated 2026-11-16 (10:12, **P3.17 — the decode's error is 100 % INHERITED: every one of the 36
convs is faithful on a good input (mean 0.14 pt of cosine, worst 0.46), and layer 34 is
exonerated (0.09 pt intrinsic vs 15.18 pt inherited).** Host-only, **no device run and no device
rate touched**: `vae_sim.py --oracle` (new arm — each conv fed the reference's own input at the
program's `SA_in`), capture `vae_sim_20261116_100611`. **Validator: `wire` reproduces P3.16 to the
digit** (20.5876 dB / 0.888501 / layer 34 0.714915); **`oracle` = 34.34 dB / 0.995595**, worst site
`conv_out` 0.995406, **0/36 sites lose > 1 pt** on a perfect input. Inherited loss climbs
1.47 pt (layers 0–11) → 4.35 (ub1) → 6.00 (ub2) → **11.03 (ub3)** while every site's own rounding
stays ≤ 0.46 pt ⇒ **no per-layer quantisation tweak can recover the decode**; what is left is the
glue (norm+SiLU requant, add, gather requant), which this instrument does not yet price. **Second
finding, a hole in every table since P3.14: layers 11 and 20 (`upsample3d.time_conv`) printed
`SHAPE MISMATCH` in all arms — 2 of 36 convs were never scored.** Cause measured:
`_decode` runs ONE latent frame at a time and `WanResample.forward` **skips `time_conv` on frame 0**
(`"Rep"` sentinel: replicates the block input, no conv), then calls it one frame at a time with a
zeroed cache, so its hooked input/output are **T−1 frames** (2/3, 4/5) — which is also why P3.14's
gather mapping is asymmetric. The tool now rebuilds those two layers' input from the upsample
block's input and its target from the same module in whole-sequence form: **36/36 convs scored for
the first time** (11: 0.986019 wire / 0.996160 oracle; 20: 0.961396 / 0.997462). The
whole-sequence-vs-streaming `time_conv` difference at frame 0 is **reported, not applied**. `T_V`
8.908 s, `T_N` exact 0.8905 s/block, `k_b*` 165 untouched. → `docs/device/RESULTS_vae_oracle.md`.*
**NEXT ACTION, in order: (a) the GLUE one-hop arm** — the same instrument extended to
`NORMSILU` / `ADD` / gather: feed each glue op the reference's own input and score its output, so
34.34 → 20.59 dB decomposes into conv vs glue and the amplification in the last up_block gets a
per-op number (host-only, one walk); **(b)** the `post_quant_conv` branch (+7.32 dB, edits the
artifact) still waits on the student's call; **(c)** the deliverable side — `proc_b --decode 1` is
still a no-op stub, then Gate 5.

*Updated 2026-11-16 (09:35, **P3.16 — layer 34 is NEITHER its residual add's grid NOR the
`conv_out` 96→3 shape: both priced, together worth 0.003 dB.** Host-only, four simulator
arms, **no device run and no device rate touched**: `vae_sim.py --layer34` → arms
`wire` / `add34` (the residual add after conv 34 summed in float32, no int8 hop) /
`coutpc` (three per-output-channel output scales on `conv_out`) / `both34`, capture
`vae_sim_l34_20261116_092942`. **Validator: `wire` reproduces P3.15 to the digit**
(20.5876 dB / 0.888501 / worst layer 34 0.714915). **`add34` +0.0016 dB, `coutpc`
+0.0015 dB, `both34` +0.0031 dB — and layer 34 stays at 0.7149 in all four**, i.e. the
worst layer does not move at all. Two corrections measured: **P3.11's "4.76×" is the add
after conv 32, not 34** — layer 34's ratio is **4.18×** (0.33601 / 0.080382) and the range
across all 14 sites is **1.39–6.13×**, with clamp never above **0.52 %**; and
**`conv_out`'s three channel maxima are 1.1442 / 1.0510 / 1.2086 = a 1.15× spread**, with
the single `sa_out` 0.0112168 calibrated 18 % above the largest, so per-channel has
nothing to win — and a trap: **per-channel requant with scalar dequant is −0.11 dB**
(20.4792 vs 20.5891), the gain and the cost being the same size. What is left at layer 34
is inherited-vs-intrinsic, which needs the oracle-input probe. `T_V` 8.908 s, `T_N` exact
0.8905 s/block, `k_b*` 165 untouched. → `docs/device/RESULTS_vae_l34.md`.*
**NEXT ACTION, in order: (a) the ORACLE-INPUT arm** — feed each conv the reference's own
input quantised at the program's `SA_in` and score its output; intrinsic-bad (~0.85 at
layer 34) ⇒ the site, input-inherited (~0.99) ⇒ upstream. Host-only, one walk, and it is
the last thing that can attribute the decode's remaining ~18 dB at all; **(b)** the
layer-11/20 3-frame-vs-2-frame `time_conv` support; **(c)** then `proc_b --decode 1` →
`convwire`, and Gate 5. The `post_quant_conv` branch (+7.32 dB, edits the artifact) still
waits on the student's call.**

*Updated 2026-11-16 (08:44, **P3.15 — the accumulating scheme is priced, and it is NOT the
missing 18 dB; the absent `post_quant_conv` is worth +7.32 dB.** Four simulator arms
(bias placement × requant granularity) plus the missing op, **host-only, no device run**:
`vae_sim.py --schemes all --pqc`, capture `vae_sim_schemes_20261116_083315`. **`wire`
scores 20.59 dB / cos 0.8885 against the device's measured 20.44 / 0.8967 — the simulator
tracks the phone to 0.15 dB**, which is what makes the arms device-comparable. Bias on
the first applied tap only: **20.30 dB (−0.29)** even though it is better per layer nearly
everywhere (conv_in 0.9959 → **0.99995**, first divergence layer 3 → 12, layers ≥ 0.99:
4 → 11) — **layer 34 collapses 0.7149 → 0.6325** and the end product falls, i.e. the
`kT`-counted bias is *cancelling* a real defect at the last up_block. Requant once
instead of per tap: **20.63 dB (+0.04)** — free and negligible. **`post_quant_conv`
(16→16, absent from the artifact): 12.33 → 19.65 dB against the same fp32 target =
+7.32 dB**, 25× the other two combined, worst clamp 5.91 → **0.43 %**; fp32-vs-fp32 the
op moves the decode **12.06 dB / cos 0.179**. **Measured, not applied — folding it edits
the artifact, so it is the student's call**, and its layer-0 input scale here is a
one-sample `max|x|/127`, not a `calibrate.py` sweep. **Layer 34
(`decoder.up_blocks.3.resnets.2.conv2`) is the worst layer in all five arms** — the named
next defect site. `T_V` 8.908 s, `T_N` exact 0.8905 s/block, `k_b*` 165 untouched. →
`docs/device/RESULTS_vae_schemes.md`.*
**NEXT ACTION, in order: (a) layer 34 — `up_blocks.3.resnets.2`: price its residual add
at `post_add_sa` (P3.11 measured the post-add range to 4.76× conv2's own) and the
`conv_out` 96→3 shape with the simulator, no device needed; (b) the layer-11/20
3-frame-vs-2-frame `time_conv` support; (c) then `proc_b --decode 1` → `convwire`, and
Gate 5. The `post_quant_conv` branch waits on the student's call before any artifact
edit.**

*Updated 2026-11-16 (08:16, **P3.14 — the decode's glue is localized and the temporal
shuffle was WRONG: fixing it takes the device decode from 11.60 dB to 20.44 dB
(+8.84 dB) for a one-line index change.** P3.13 named the instrument (a host program
simulator); `tools/npu/vae_sim.py` is it, and it found the defect on its first walk.
It runs `vae_wire.txt` op-for-op with the device's semantics — int8 codes at the declared
per-tensor scales, the per-tap split with each tap rounded to int8 before the sum, the
INT32 bias in accumulator units, the same `g = act.sa/sa_in` requant — from the dumped
latent the device consumed, and diffs each conv against the fp32 reference's **own**
module output (hooked, chunk-concatenated, asserted to the plan's 3/5/9 frames). Layers
0–10 track at **0.965–0.996** (so `NORMSILU` and `ADD` are exonerated), then layer 12
collapses to **0.717**. The mapping was then MEASURED off the reference's own `resample.1`
conv input: frames are **[b0h0, b1h0, b1h1, b2h0, b2h1]** (0.896/0.977/0.972/0.970/0.963)
where the wire emitted both halves of `b[0]` and dropped `b[T-1]`'s last half — same
frame COUNT `2T-1`, so it failed silently, with no shape error and *better* saturation.
Fix in `npu_block.cpp`: `t==0 ? (0,0) : ((t+1)/2, (t+1)%2)`. Measured: gather 0.6114 →
**0.9536**, layer-12 conv 0.7172 → **0.9638**, device decode vs fp32 **(A) 11.60 dB /
cos 0.2190 → 20.44 dB / 0.8967**, B 20.40/0.8946, C 20.42/0.8948. Device run
`vae_wire_fix_20261116_080721`: 36/36 convs, 592/592 executes, `pc=1` on all 36, 0
fallbacks, compile 18.81 s, wall 44.932 s, clamp 0.0149 → **0.0137 %**; MECHANISM-GRADE
(TPU 24.0 vs gate 23.6, 0.4 °C over) so **no rate is claimed — `T_V`/`T_N`/`T_G`/`k_b*`
untouched**. The same fix carries through the **second** upsample (layer 21: gather
0.9445, conv 0.9488) and **no further collapse exists between layers 12 and 21** — the
error now drifts 0.996 → 0.94 instead of stepping, i.e. what an accumulating scheme cost
looks like. → `docs/device/RESULTS_vae_glue.md`, tool `tools/npu/vae_sim.py`.*

*Updated 2026-11-16 (07:36, **P3.13 — the device decode has a quality number and it FAILS:
11.60 dB / cosine 0.219 against the fp32 reference from the identical quantised latent. Two
defects came out of it. (1) FOUND AND FIXED — the program's tap→frame labels were
**mirrored**: `WanCausalConv3d` pads 2 zeros at the FRONT, so weight slice 0 is applied to
`x[f-2]`, but `vae_export` labels slice 0 `"f"`; emulated against the checkpoint on
`decoder.conv_in` the labelled pairing scores cosine **0.0558** and the mirrored one
**1.00005**, so all 36 convs had been reading the wrong frames. (2) The HAL is EXONERATED on
`CONV_2D`: with the fix, layer 0's device codes are **98.59 % bit-identical** (100 % within
±1, cosine 1.000312) to the exact emulation of the program — unlike `FULLY_CONNECTED` in
Gate 3 — so the remaining error is the **glue** (NORMSILU / ADD / SAVE-LOAD / shuffle
gathers), not the conv. Also measured: each tap's graph carries the same INT32 bias (bias
counted kT times) and each tap is rounded before the sum, costing ~1 % cosine per layer;
and `post_quant_conv` (16→16, non-identity) is **missing from the program entirely**.
Mechanism-grade thermal (24.0 vs 23.9 gate). → `docs/device/RESULTS_vae_psnr.md`, tools
`tools/npu/vae_psnr.py` (new), `tools/npu/conv_parity.py` (new), `npu_block convwire … 1 [L]`
(dump + per-layer probe), captures `vae_wire_dumpr_20261116_073415`,
`vae_dump_layer0_fix_20261116_073615`, `vae_psnr_20261116_073602`,
`conv_parity_20261116_073621`.*
*Updated 2026-11-16 (07:02, **P3.12 — the decode is WIRED and it executes: 36/36 convs, 592/592
`CONV_2D` executes, and the saturation that P3.10 could only point at (worst layer 38.57 %) is
**0.0721 %** with the norm in place — 19/36 layers clamp nothing.** `npu_block convwire` walks a
linear program (`OP CONV/NORMSILU/NORM/SILU/SAVE/LOAD/ADD/SKIP`) derived in EXECUTION order by the
new `tools/npu/vae_wire.py`, so every conv carries its own `artifact_index` (`forward` applies
`conv_shortcut` FIRST — 3 positions were being read out of order) and every conv's OUTPUT operand
now carries its measured `SA_out` instead of the consumer's scale (the two span 0.275…15.5×).
Measured: 100 taps, 101 program ops, `pc=1` on all 36, 0 fallbacks/failures/fatals; **sum of
execute walls 9,056.3 ms** (15.30 mean, 44.56 max) vs P3.10's 8,822.6 (+2.6 %, and this walk heated
24→39 °C against the decode's +1.7 % temp term); whole walk 44.86 s, compile 18.79 s. **The wiring
costs more than the convs**: 10,043.5 ms norm+SiLU (1.238 G elements) + 1,070.1 adds + 1,195.6
epilogue + 630.7 prep — a scalar C++ reference pass (the norm is per-pixel, so it cannot fold into a
conv epilogue), which is why `T_V` = 8.92 s still holds only if it is fused. **MECHANISM-GRADE:
started 0.4 °C over the gate (24.0 vs 23.6), no rate claimed.** Defect paid for: `vae_wiring.json`'s
`gamma_off` is in **BYTES** (P3.8a's trap again) — caught by the device's own range check on the
first full attempt. **No quality number yet: it executes in the right range, it does not yet
decode.** → `docs/device/RESULTS_vae_convwire.md`, capture `vae_wire_20261116_070210`.*

*Updated 2026-11-16 (06:37, **P3.11 — the decode's wiring is measured, and the chain between two
convs is NOT a rescale: it is a `WanRMS_norm` + SiLU.** `calibrate.py` now records the real per-layer
OUTPUT range (the old `max_abs_out` was the whole decode's output, a flat 1.0) plus the POST-ADD range
per resnet; `act_scales2_f32.bin` = 2×36 f32 (SA_in ++ SA_out), and `vae_wiring.json` +
`norms_f32.bin` carry the 128-op plan and all 29 norm gammas. The 89× step between
`mid_block.resnets.0.conv1` (max 54.79) and `.conv2` (max 0.612) is **proved** to be
`F.normalize(dim=1)*sqrt(C)*gamma` then SiLU: hand-applied, it reproduces the consumer's measured
input range to **0.00 %**. Also recorded: the artifact's layer index is a storage order, not an
execution order (shortcut first in `forward`, last in the artifact). Host-only, no device number
touched — the norm/SiLU/shuffle/add are still unexecuted. → `docs/device/RESULTS_vae_wiring.md`.*

*Updated 2026-11-16 (06:10, **P3.10 — the measured per-layer activation scale is wired into the
device: `convseq` now runs all 36 layers at their own `SA` (was `1/127` for every one), with real
INT32 per-channel bias and the requant to the *consumer* layer's scale. 592/592 executes,
`sa=table` accepted on 36/36, and the wiring costs **−0.14 %** (8,822.6 vs 8,834.7 ms) — so it is
free. Saturation is counted now: 17/36 layers none, median 0.63 %, worst 38.57 %, and it is not the
scale ratio (r = 0.50) but the still-missing residual/SiLU path.** Before that, **P3.9 — `SA` is now measured, not declared: 1800 DiT scales
(6 locked timesteps × 30 blocks × 10 GEMMs) + 36 VAE layer scales, and with **both operands INT8**
the decoder still PASSes Gate 1 at **38.73 dB / 0.9672** against 40.22 dB weight-only ⇒ the
activation path costs **1.48 dB** (`phase1_quant/calibrate.py`,
`docs/device/RESULTS_calibrate.md`, `docs/device/RESULTS_gate1_actquant.md`).** Before that,
**P3.8c — the VAE decode at a table-grade start: 592/592 `CONV_2D`
executes in 8,834.7 ms (0.837× the plan's per-op sum) from a 24.0 °C start against a 24.1 °C gate,
and the decode's `CONV_2D` class carries its own temperature term (+1.7 % median, +5.2 % worst,
identical shapes early→late) that the FC stage does not; and **Gate 1's end-to-end PSNR/SSIM leg
PASSES at 40.22 dB / SSIM 0.9708** weight-only, with the two 0.9974-cosine `time_conv` layers
carrying the whole cost). This
file is the orders sheet: read it first, then `docs/00_START_HERE.md`. The pipeline detail is
`docs/device/RESULTS_stream_p35.md` (run `stream_20261116_000415`); Gate 4 is
`docs/device/RESULTS_gate4_p36.md` (runs `gate4_20261116_003217` / `…_003411`); Gate 3 is
`docs/device/RESULTS_gate3_parity.md` (captures `gate3_parity_20261116_005432` / `…_010101`), its
bisect is `docs/device/RESULTS_gate3_bisect.md` (captures `gate3_bisect_20261116_012157` … `_012349`),
and the cost of the exact recipe is `docs/device/RESULTS_tn_recipe.md` (4 runs,
`tn_recipe_20261116_013019` … `_013625`).*

## Mission

Reproduce HeteroVDiff's design on a **Pixel 6a (Tensor G1)**: one INT8 copy of DCM-Wan2.1-1.3B's
base weights shared by two engines, a single block-level cut of the `B = 180` denoising
trajectory, and a two-stage chunk pipeline. Course deliverable = working pipeline + measured
tables + IEEE report; the student owns the hardware rig, slides and paper.

## Where things stand

| | |
|---|---|
| Repo | `C:/Users/harri/Documents/GitHub/HeteroVDiff_Pixel_6a` |
| Device | Pixel 6a attached, serial `25311JEGR11811`, shows as `device` |
| Weights | `weights_raw/DCM_WAN/` — transformer 5.73 GiB FP32, VAE 508 MiB, umT5 5 shards |
| Checkpoint manifest | `configs/block_manifest.json` (verified against the paper) |
| Gate 0 (capabilities) | **done** → `docs/device/RESULTS_gate0.md` |
| Gate 2 (zero-copy) | **pass** on both routes; file-mmap route B preferred |
| **Mali INT8 GEMM** | **done** → `docs/device/RESULTS_gemm_mali.md`; 161.8 ms/block, 877 GOPS, exact vs int32 reference |
| **NPU block graph (`T_N`)** | **measured as a sum** → `docs/device/RESULTS_npu_tn.md`; 9 GEMMs gate-compliant (0.4178 s) + `ffn.down` chunked 7×K=1280 (0.2580 s) = **0.6758 s/block**, 210 GOPS |
| **VAE decode (`T_V`)** | **both branches measured** → `docs/device/RESULTS_vae_tv.md`; TPU (36 causal convs as `kT` shifted `CONV_2D` taps) **10.554 s/chunk**, 969.7 GOPS; Mali (`conv8_taps`, implicit GEMM, one dispatch per class) **12.256 s/chunk**, 417.5 GOPS, 7/13 classes exact vs the int32 reference. **TPU 1.16× faster** on the convs — the decode ordering is the opposite of the block's. Mid-block attention (32.4 GMAC) **now measured on both engines**: TPU **0.601 s**, Mali **0.0855 s** (one dispatch per GEMM) ⇒ **Mali 7.0× faster**, the opposite ordering again. `T_V` = **11.155 s all-TPU · 10.640 s hybrid · 12.342 s all-Mali** — no longer a lower bound. The decode now also **executes** end-to-end off the INT8 artifact: **8,834.7 ms** of conv compute, 592/592 dispatches, table-grade start → `docs/device/RESULTS_vae_convseq.md` §P3.8c. **P3.10 (06:10) then wired the measured per-layer `SA` in: 592/592 at `sa=table`, sum of execute walls 8,822.6 ms (−0.14 % ⇒ free), saturation counted (17/36 layers zero, worst 38.57 %)** → §P3.10. **P3.12 (07:02) then EXECUTED the wiring itself: 36/36 convs, 592/592 executes, `pc=1` on all 36, and saturation collapses from 38.57 % worst to **0.0721 %** (19/36 layers clamp nothing) now that the norm and the add-at-`post_add_sa` are real; the glue costs 10.04 s of norm+SiLU + 1.07 s of adds against 9.06 s of conv, so `T_V` still holds only if it is fused** → `docs/device/RESULTS_vae_convwire.md` |
| Gate 1 (artifact) | **INT8 base + verifier done** → `docs/device/RESULTS_gate1.md`; 1,391,984,640 B INT8 (exact), 300/300 GEMMs cosine ≥ 0.999922, coverage 1317/1317 tensors. **End-to-end PSNR/SSIM leg now PASS** (2026-11-16 05:10, `phase1_quant/psnr_gate1.py`, capture `gate1_psnr_20261116_051022`): **40.22 dB / SSIM 0.9708** with all 36 decoder layers rebuilt from the INT8 artifact vs the fp32 reference (bar 28 dB / 0.92), 36/36 reconstructions reproduce the manifest's `rel_fro` to 4.5e-07; leaving the two 0.9974-cosine `time_conv` layers fp32 recovers **48.99 dB / 0.9962** ⇒ those two layers are the whole cost. **Weight-only** — and now also measured **with the INT8 activation operand** at the calibrated per-layer `SA`: **38.73 dB / 0.9672 PASS**, i.e. the activation half costs 1.48 dB / 0.0036 SSIM → `docs/device/RESULTS_gate1_actquant.md`, capture `gate1_psnr_20261116_054851` |
| **Gate 3 (parity)** | **measured, FAILS as configured — but LOCALIZED and a passing recipe exists** → `docs/device/RESULTS_gate3_parity.md` + `docs/device/RESULTS_gate3_bisect.md`, tools `tools/pipe/gate3_parity.py`, `tools/pipe/gate3_bisect.py`, `proc_b --dump-parity/--probe-dir/--act`. The break is the **M (row) direction**: `N = 384` graphs are **100.00 % bit-exact (Δ = 0, 0 elements off by ≥2)** at `M = 1/8/128/160/192/200`, `N = 768` at `M = 128`, `N = 1536` at `M = 1`, on both `attn1.to_q` and `ffn.up`; the same 384-column graph at `M = 384` **will not compile** (`OP_FAILED`, ~58 ms). Wide graphs at `M = 256…384` return 25.8–50.7 %. So the fix Gate 3 proposed ("one graph per 384-column block") is the exact graph this compiler refuses, and the exact recipe is `N ∈ {128,384,768}` × `M ≤ 200` = **112 executes/block** instead of 9 at 384 tokens (**448 executes / 56 compilations at the stage's 1536 tokens; cost now measured in P3.7c**) |
| Gates 4–5 | Gate 4 PASS (memory + one copy) |
| **Exact recipe `T_N` (P3.7c)** | **measured** → `docs/device/RESULTS_tn_recipe.md`: the only bit-exact geometry (P3.7b) costs **0.7164 s/block** for the nine GEMMs = **1.90× the 377.8 ms 9-graph stage**, **140 GOPS** vs 265, at **56 compilations (8.64 s) / 448 executes** per block — and returns **448/448 tiles bit-exact (Δ = 0 LSB)**, `ffn.up`'s tail-128 block included, checked byte-wise against the artifact. 4 runs, 0.3 % spread. `solve_cut.py` on the measured pair: `k_b* = 166`, period **24.47 s (+3.6 %)** — **reported, not applied** (solo vs the locked co-run number) |
| **Exact recipe, co-run (P3.7d)** | **measured** → `docs/device/RESULTS_corun_recipe.md`, capture `corun_recipe_20261116_020926` (pilot `…_020535`), tool `tools/corun/co_recipe.py`. Under GPU load the recipe costs **700.5 ms/block** vs **707.0 ms** before the GPU joined in the same run (−0.9 %) ⇒ **contention is not measurable on either engine**; GPU block **146.05 ms** while the recipe executes vs **160.27 ms** solo (same 8–9 % "faster under load" direction as `RESULTS_corun_tg.md`, cause still DVFS-unestablished); overlap **100.0 %** from CLOCK_MONOTONIC stamps. Cost of the **pass-major** rewrite (needed so a pass is classifiable): all **56 graphs held at once**, compile **9.57 s/block**, 448 executes/pass, solo sanity **710.3 ms**. `T_N` = 0.7005 + 0.2580 `ffn.down` (**still solo**) = **0.9585 s** ⇒ `k_b* = 166`, period **24.25 s (+2.7 %)** — **reported, not applied** |
| **Co-run `T_G` / `T_N`** | **measured** → `docs/device/RESULTS_corun_tg.md`; both engines executing at once, 99.8 % of GPU-time overlapped NPU executes. GPU base block **0.1474 s** (no slowdown — it measured 8 % *faster* than solo), NPU 9-GEMM round **0.3904 s** (±4 % per GEMM). `T_N/T_G = 4.40×` |
| **Sustained `T_N`** | **measured** → `docs/device/RESULTS_sustain_npu.md`; 60 rounds × 9 GEMMs = **24.6 s of continuous TPU work, drift +0.35 %** (first-20 % 408.9 / last-20 % 410.3 ms per round, slope +0.03 ms/round) while the TPU zone rose 25.0 → 31.0 °C ⇒ **no derating over a full NPU stage**, so `T_N` carries no throttle term. Tool: `python tools/npu/sustain_npu.py` |
| **`k_b*`** | **solved = 160** → `configs/rates_preset_a.json` + `phase2_profile/solve_cut.py`; period 23.61 s, 5.394× vs NPU-only, GPU `[0,160)` + NPU `[160,180)` + decode. Survives the attention term **only if the attention runs on Mali** (hybrid decode, `T_V` = 10.640 s); all-TPU decode moves it to 161 |
| **Gate 4 (memory + one copy)** | **PASS twice** → `docs/device/RESULTS_gate4_p36.md`, tool `tools/pipe/gate4.py`, captures `gate4_20261116_003217` / `…_003411`. PSS A+B peak **282.8 MB (0.27 GiB) vs the 2.2 GiB budget — 8.2× headroom** (A 246.9 MB median, B 30.9 MB). `proc_a`: **one `rw-s` MAP_SHARED view** of the 46,399,488 B block at offset 0, **10/10** windows imported by reference. `proc_b`: one fd, handed to `createFromFd`, **0 mappings** of its own. **Trap: `smaps` under-reports the GPU stage 2.93×** (`Pss` 85,137 kB vs `dumpsys` TOTAL 249,290 kB — the 164,192 kB `GL mtrack` row is driver-accounted CL memory and is absent from `/proc/<pid>/smaps`) ⇒ quote `dumpsys`, never `smaps`, for Mali |
| Blocker | every rate the cut needs is now measured on both engines **and checked for sustained drift** — the mid-block VAE attention closed 2026-11-15 21:2x, the 24.6 s sustained soak 21:41. What is still unrun: **nothing host-only** — the Gate-1 PSNR/SSIM leg and `calibrate.py` both landed 2026-11-16 (40.22 dB weight-only, 38.73 dB with the INT8 activation operand; 1800 + 36 measured scales) — **environment resolved 2026-11-15 (commit `8663c3f`): use `.venv-ref/Scripts/python.exe`** (torch 2.15.0+cpu, diffusers 0.43.0, transformers 5.23.0, deliberately NOT `.venv-quant` because this cron fires every 15 min in this tree). Measured fp32 GEMM 4096³ = 449.9 GFLOPS, so a 36.0 TFLOP 9-frame reference chunk is ~1.3 min. **Gate 4 is built and PASSing**; what remains of Gates 3–5 is Gate 3's numerical parity, the decode as an execution, and Gate 5 |

## The two results that changed the project

The tiled Mali kernel runs a Preset-A DiT block in **161.8 ms** (877 GOPS) against the Edge TPU
arithmetic bound of **532 ms**. The GPU is **3.3× faster per block** than the TPU, not 31.6×
slower as the naive kernel suggested. The paper's premise (NPU ≈ 3.4× faster than GPU, therefore
NPU takes the suffix) **inverts on Tensor G1**. Per the locked policy this is reported, not
tuned: `k* = (B·T_N + T_V)/(T_N + T_G)` now lands late, so the GPU takes nearly the whole
trajectory and the NPU's real job becomes covering the VAE decode.

The TPU side then got measured (2026-11-15, `RESULTS_npu_tn.md`) and is **worse than the
arithmetic bound, for structural reasons**: 9 of the block's 10 GEMMs run at 204–289 GOPS
(`ffn.up` 36.62 ms/tile = 289 GOPS), summing to **≥ 0.4178 s/block — already ≥ 2.58× `T_G`** for
70 % of the MACs. Three hard limits: a whole-block single graph **reboots the phone** in
`Compilation_finish`; `ffn.down` (K=8960) **never compiles** (`DEAD_OBJECT` after ~62 s hot,
`OP_FAILED` after 47 s cool); and
**per-output-column weight scales are rejected by `FULLY_CONNECTED`** at `addOperation`, so the
locked quantisation cannot be expressed on the NPU stage.

`ffn.down` was then closed (2026-11-15 17:48, `RESULTS_npu_tn.md` §4): the compiler's limit is the
**reduction dimension** — K ≤ 1664 compiles in ~0.5 s, K ≥ 1792 grinds for 46–59 s and returns
`OP_FAILED` — so the down projection has no graph at K=8960 *or* at any 2-way/4-way split. It does
run as 7 × FC`[1536,1280]` + 6 × `ADD`, one compilation per op (2.57 s of compile), at
**0.2580 s/block (163.9 GOPS) — 5.45× the Mali time for the same shape (47.33 ms)**. `T_N` is
therefore **0.6758 s/block** and `T_N/T_G = 4.18×`: the GPU is 4.2× faster per block, and the down
projection belongs on Mali whichever way the cut lands. Both branches are measured; the choice is
the student's.

Caveat that must survive into the paper: 161.8 ms and the NPU per-GEMM rates are **solo** kernel
rates, and a cool start is **1.6 % slower** than the earlier hot ones, i.e. nothing was throttled.
The co-run then closed the gap the cut actually needs (`RESULTS_corun_tg.md`): with both engines
executing at once the GPU base block costs **0.1474 s** and the NPU's 9-GEMM round **0.3904 s** —
neither engine loses its solo rate (the GPU measured 8 % *faster* under load). `T_G`/`T_N` from the
co-run give **`k_b* = 160`** (period 23.58 s, 5.40× vs NPU-only). Reaching the idle floor took 17 min
with the device network off — use `python tools/npu/cool_rerun.py`, never a bare timed run.

## Ordered next actions

1. ~~**Cool re-run of `T_N`**~~ — **done 2026-11-15 16:00:42**: capture `…_160146`, TPU 29.0 °C start
   against a 29.1 °C gate, gate-compliant table in `RESULTS_npu_tn.md`. Reusable tool:
   `python tools/npu/cool_rerun.py` (waits for the floor, then runs the harness under it). Note:
   with the Play Store background service running the TPU zone sat at 44–48 °C and the gate was
   unreachable — kill device network for the run.
2. ~~**`ffn.down` path**~~ — **done 2026-11-15 17:48, both branches measured**. Single FC has no
   compilable graph at K=8960/4480/2240/1792 (compiler K ceiling: 1664 OK in ~0.5 s, 1792 fails
   after 48.7 s). Measured: Mali **47.33 ms/block** vs Edge TPU 7×K=1280+6×ADD
   **258.0 ms/block** ⇒ 5.45× in Mali's favour. New modes: `downsep`, `kcompile`, `ksweep`,
   `diag` (see `RESULTS_npu_tn.md` §4). `T_N` = **0.6758 s/block**.
3. **Per-column scales vs the NPU stage — cost now measured, decision still yours.** `FULLY_CONNECTED`
   takes per-tensor weight scales only. Measured on the real weights (40 GEMM instances, blocks
   0/5/15/29): per-tensor costs **2.15× the mean GEMM output error** (3.01 % vs 1.40 %, worst 5.44×)
   and fails the Gate-1 `cosine ≥ 0.999` weight bar on 5/40 instances (worst 0.996892) where
   per-column fails none; the penalty tracks column-scale spread at **r = 0.805**, i.e. it is a
   weight property, not a shape property — so "restrict the NPU to shapes that quantise the same"
   is not supportable, and the cheap subset (`attn2.to_k`+`to_v`) is only **10.2 % of block MACs**.
   → `docs/device/RESULTS_scales.md`, `phase1_quant/scale_parity.py`. Touches `design_locks.json`.
4. **`T_V` — VAE decode.** ~~TPU branch done~~ **both branches now measured 2026-11-15**:
   TPU `CONV_2D` taps **10.554 s/chunk** (969.7 GOPS, gate-compliant start) vs Mali
   `conv8_taps` **12.256 s/chunk** (417.5 GOPS, TPU 26.0 °C start, 7/13 classes exact) ⇒
   **the NPU is 1.16× faster at the decode**, the opposite ordering to the DiT block.
   Still open: (a) the mid-block 3-D attention (32.4 GMAC, 0.6 % of MACs) — no measured
   rate on either engine, so `T_V` is a lower bound both ways; (b) the Mali conv runs at
   48 % of its own GEMM rate (378–486 vs 877 GOPS) because the activation tile is re-staged
   per tap — a halo-staged tile is the next measurement, not a model.
5. ~~**Co-run `T_G`**~~ — **done 2026-11-15 19:48** (`RESULTS_corun_tg.md`, capture `corun_tg_20261115_194809`):
   two processes, one `adb shell`, 99.8 % overlap proven from `CLOCK_MONOTONIC` stamps. GPU base block
   **0.1474 s** co-run vs 0.1618 solo (it got *faster*, not slower — contention below the noise floor),
   NPU 9 GEMMs **0.3904 s** co-run (±4 % per GEMM vs solo) ⇒ `T_N` co-run = 0.6484 s (downsep part still
   solo). New tool: `python tools/corun/co_run.py` (`npu_block corun` holds 9 compilations and loops).
6. ~~`python phase2_profile/solve_cut.py --Tg … --Tn … --Tv …` → commit the measured `k_b*`~~ —
   **done**: `configs/rates_preset_a.json` carries the measured pair; `k_b* = **160**` (period 23.58 s,
   5.40× vs NPU-only; 158 on the conservative solo pair, 162 with the Mali decode). The cut is sharp —
   step-level `k_b` (multiple of 30) costs 27 %, and dropping `T_V` from the solve costs 35 %.
7. ~~Phase 1 (quantiser + Gate 1)~~ — **INT8 base artifact done 2026-11-15 20:5x**:
   `phase1_quant/quantize_int8.py` (40.1 s) + `phase1_quant/verify_artifact.py` (61 s, independent:
   re-reads the bytes, re-quantises the source, demands byte identity). 30 regions × 46,399,488 B =
   **1,391,984,640 B INT8, delta 0** vs Gate 1; 44.25 MiB/region (limit 47.29); MAC-weighted weight
   error **0.93 %**, cosine 0.999956, worst GEMM `ffn.net.2` 1.25 %/0.999922 — all 300 ≥ 0.999;
   max saturation 0.070 %; FP16 remainder 54,024,320 B over 525 tensors (0.028 % worst); coverage
   1317/1317. → `docs/device/RESULTS_gate1.md`. Bug caught en route: a suffix filter had dropped all
   30 per-block `scale_shift_table` (276,480 params) while still printing PASS — classification is
   now by role, with a coverage assertion in both scripts.
   **Still open in Gate 1:** (a) the end-to-end FP32-vs-INT8 PSNR ≥ 28 dB / SSIM ≥ 0.92 leg — the
   environment for it now exists (`.venv-ref`, see the blocker row above: torch 2.15.0+cpu,
   diffusers 0.43.0, transformers 5.23.0, installed 2026-11-15 in commit `8663c3f`), so this is
   code to write, not a wall; (b) `calibrate.py` (deliverable 4) — same situation, it has to *run*
   the FP32 forward pass to record `max|x|/127` per (step, block, GEMM).
   **Done 2026-11-15 20:5x, host only** → `docs/device/RESULTS_phase1_artifacts.md`:
   deliverable 2 `phase1_quant/expert_factors.py` (`artifacts/experts/detail/`, **240,089,216 B FP16,
   delta 0** vs Gate 1; 492 tensors = 30 × 16 LoRA + 12 timestep; FP16 worst relF 0.0220 %; writing
   0 INT8 bytes and asserting disjointness with the 300 base GEMMs, so the one-copy rule holds;
   `routing.json` = 180 block executions, semantic×3 + detail×3) with an independent verifier
   `phase1_quant/verify_experts.py` (re-implements the classifier, re-derives every byte → PASS);
   and deliverable 3 `phase1_quant/vae_export.py` → `configs/vae/{layers,plan_npu,plan_mali,
   scaling,tiles}.json` (37 ops, 5149.68 GMAC **MATCH** vs `vae_decode_ops.json`; binding the measured
   per-class rates back in reproduces **10.554 s** TPU / **12.256 s** Mali exactly; 592 CONV_2D
   executes vs 216 `conv8_taps` dispatches). Two facts recorded there, not fixed: `lora.scaling_alpha`
   is **null** (no alpha tensor exists in the checkpoint — copy it from the reference repo, do not
   guess), and the spec's 60×40-latent-pixel tile **degenerates to one tile** at Preset A (latent
   plane is 34×60).
8. ~~**Mid-block VAE attention**~~ — **done 2026-11-15 21:2x, both engines measured.** New modes:
   `npu_block attn` (`tools/npu/npu_host.py --modes attn --out attn_tv`) and `gemm_bench gemm <M> <N> <K>`
   (`tools/gemm/attn_mali.py`). 6120 tokens × 384 ch = 2× `[6120,6120,384]` (QK^T, AV) + 4× `[6120,384,384]`.
   TPU **0.601 s/chunk** (score 15.656 ms per M=384 tile, 115 GOPS; proj 1.568 ms, 72 GOPS);
   Mali **0.0855 s/chunk** (one dispatch per GEMM: 36.071 ms + 4×3.337 ms, 804 GOPS) ⇒ **Mali 7.0× faster**.
   `T_V` is no longer a lower bound: **11.155 s all-TPU · 10.640 s hybrid · 12.342 s all-Mali**, and
   `k_b*` = 161 / **160** / 162 — the locked 160 survives with the attention on Mali. Trap paid for:
   the Mali tiled kernel writes past `C` when N is not a multiple of 64 (`-14` on the completion read,
   while printing 10,502 GOPS); N padded 6120 → 6144. → `RESULTS_vae_tv.md` §7.
9. ~~**Sustained NPU soak (throttling over a full stage)**~~ — **done 2026-11-15 21:41:49**, capture
   `sustain_npu_20261115_214149`: 60 rounds × 9 GEMMs = **24.6 s** of continuous TPU work (longer than
   the stage's 23.6 s), per-round 408.902 ms first-20 % vs 410.346 ms last-20 % = **+0.35 %**, slope
   +0.03 ms/round, worst single GEMM `attn2.to_v` +2.2 %; TPU zone 25.0 → 31.0 °C and nothing derated.
   Start was exactly on the `battery + 3` line (25.0 = 25.0), so it is table-grade. **`T_N` takes no
   throttle term.** Not covered: `ffn.down` under contention, and the decode's `CONV_2D` class was not
   soaked for 23.5 s. Tool: `python tools/npu/sustain_npu.py --rounds 60`.
10. **NEXT ACTION (Phase 3, Gates 3–5).** Every rate the cut needs is now measured and drift-checked; build the
   two-process pipeline that streams ≥10 chunks on the phone: GPU process on blocks `[0,160)`, NPU
   process on `[160,180)` + decode, one handoff per chunk, no manual intervention.
   **Build order is written down: `phase3_pipeline/PLAN.md`** (P3.1 handoff selftest → P3.2 GPU stage →
   P3.3 NPU stage → P3.4 coordinator/2 chunks → P3.5 10 chunks → P3.6 Gate 4).
   **P3.1 DONE 2026-11-15 22:03** → `docs/device/RESULTS_handoff_p31.md`: `phase3_pipeline/handoff.h` +
   `tools/android/pipe/handoff_selftest.cpp` + `tools/pipe/handoff_selftest.py`. 200 chunks × 2 configs,
   **0 mismatching words of 587,520 per chunk, 200/200 in order**, publish→consume **680.9 µs** mean
   (661.7 median / 1023 µs max) free-running, **2541.4 µs** at a 1 ms consumer stall; `msync` 4 KiB
   median **63.7 µs**; ring file exactly **9,408,512 B = 2 slots**; split `--role producer|consumer`
   launch from two adb sessions passes 30/30. Producer blocked on 198/200 chunks even free-running, so
   the 2-slot ring's `done` gate is the normal path, not an edge case.
   **P3.2 DONE 2026-11-15 22:25** → `docs/device/RESULTS_proc_a_p32.md`: `tools/android/pipe/proc_a.cpp`
   + `tools/pipe/proc_a.py` run the real artifact — mmap `block_00.bin`, 10/10 weight windows imported
   zero-copy, 80/80 GEMMs, 8/8 chunks published and drained by a second PID. **The finding: zero-copy
   costs the GPU 32 %/block** — import **214.5 ms** (662 GOPS) vs device-local staging **162.2 ms**
   (875 GOPS, independently reproducing the 161.8 ms block rate). Attention only +4 % but both FFN
   weights (13.8 MiB each) **+15/16 %**, i.e. the penalty tracks the weight working set that misses
   cache. Cut consequence reported, not applied: `k_b*` **157** (staging, period 25.55 s) / **148**
   (import, 31.75 s) vs the locked 160 / 23.61 s. Closure barrier measured **1.7 µs**; payload
   conversion is host-side and costs **44–57 ms/block** (removable with an fp16 epilogue kernel).
   **P3.3 DONE 2026-11-15 22:53** → `docs/device/RESULTS_proc_b_p33.md`: `tools/android/pipe/proc_b.cpp`
   + `tools/pipe/proc_b.py` are the NPU stage — one `createFromFd` view of `block_00.bin` with
   per-GEMM `setOperandValueFromMemory` windows, 9 held one-op graphs compiled in **4.52 s**,
   **108/108 executes**, 3/3 chunks consumed, producer/consumer payload hash chain **MATCH**.
   **The 9-GEMM stage is 377.8 ms/block (265 GOPS) — 0.967× the co-run 390.4 ms the cut uses,
   measured through the real weights and the ring; three runs 373.4–381.1 ms (2.1 % spread).** `ffn.down` both branches in the same
   harness: TPU 7×K=1280+6×ADD **272.3 ms/block** vs Mali **47.33 ms** ⇒ **5.75× slower**
   (whole NPU block 656.8 vs 388.1 ms). The TPU branch needs a 13,762,560 B weight repack
   (row-major cannot be K-sliced) — **one-copy deviation, reported not adopted**. Handoff:
   first chunk 0.163 µs, later chunks **queue** (consumer mid-block: 391–710 ms) — that is
   back-pressure, not handoff cost. Thermal: `skip` started 0.2 °C (pre-fix) and 1.0 °C
   over the floor gate (floor-grade, stated), `stage` 4.1 °C over (comparison only, not table-grade).
   **P3.4 DONE 2026-11-15 23:1x** → `docs/device/RESULTS_stream_p34.md`, tool
   `phase3_pipeline/stream.py`, captures `stream_20261115_231417` (4 chunks × 1 block) and
   `stream_20261115_231451` (2 chunks × 20 blocks = the cut's real stage). Both PASS:
   4/4 and 2/2 published→consumed, 144/144 and 1440/1440 executes, 10/10 zero-copy imports,
   no fatal. **Launch order is a correctness rule**: both binaries `reset_headers()` on map,
   so the consumer must hit `READY` before the producer starts, else a publish is wiped and
   the run hangs 300 s later as `chunk_never_published`; ordered that way the first handoff is
   **0.244 µs**. GPU block **187.2 ms** (10 closures) / **168.0 ms** (one closure), NPU block
   **385.9 ms**, overlap **39.4 % of NPU busy**. Two findings now block P3.5:
   (a) **`proc_a` publishes per block, the cut hands off once per 160** — GPU finished
   2×20-block chunks in 0.214 s and idled 18.4 s (overlap 1.1 %); needs `--pub-every N`.
   (b) the P3.3 harness FNV-hashes `ffn.up`'s 13,762,560 B output **every block** →
   reproducible **~80 ms dead gap between blocks** = 1.62 s per 20-block chunk (**17 %**),
   12.8 s at 160 blocks/chunk; sample the hash before quoting a period. Both captures start
   1.8 / 3.7 °C over the gate (back-to-back) → mechanism-grade, rates need `cool_rerun.py`.
   **P3.5a DONE 2026-11-16 00:0x** → `docs/device/RESULTS_stream_p35a.md`, capture
   `stream_20261115_233709` (2 × 160 GPU blocks, 20 NPU, table-grade start TPU 25.0 vs 25.3):
   `--pub-every` / `--hash-every` / `--once-blocks` added (defaults 1 = P3.4 behaviour), PASS
   2/2 + 2/2, 1440/1440 executes, **GPU block 202.1 ms over n=320 (no drift, −0.34 %)**; on the
   same import route the one-closure reference is **166.2 ms**; NPU block **392.8 ms** (n=40);
   publish interval **32.544 s**, publish→consume **0.20/0.45 µs**, back-pressure **0 µs**;
   `ffn.up` hash 85.5 ms per call ⇒ 170.9 ms total instead of 3.4 s. **The finding: the period
   at the locked `k_b` = 160 is 32.54 s, 1.38× the 23.61 s prediction**, decomposing into
   +3.0 s zero-copy import and +5.7 s per-GEMM closure barriers; `solve_cut.py` on the measured
   pairs gives **k_b* = 157** (`T_G` 0.1662, 26.09 s) / **150** (0.2021, 30.32 s) vs the locked
   160 — reported, not applied.
   **P3.5 DONE 2026-11-16 00:04** → `docs/device/RESULTS_stream_p35.md`, capture
   `stream_20261116_000415` (10 chunks × 160 GPU blocks + 20 NPU blocks, table-grade start
   TPU 25.0 vs 25.5): PASS 10/10 published + 10/10 consumed, **1600** GPU blocks,
   **7200/7200** executes, wall 370.4 s. Three numbers: (a) **steady-state period 32.72 s**
   (chunks 2–7, 32.61–32.81) — P3.5a's 32.544 was not a startup artifact; (b) overlap
   **90.02 % of NPU busy = the structural ceiling `(N-1)/N` for N=10**, so the coordinator is
   not what costs time — the GPU is the long pole by 4.13×; (c) **NEW: the Mali stage derates
   +35.1 % at block ~1135 (232 s of GPU work in), a step not a drift** (203.0 → 274.2 ms median,
   no recovery; NPU flat 396.5 → 394.0 ms in the same window; the post-stream one-closure pass
   measures 251.2 ms vs 166.2 cold, so it is engine-wide). Period becomes 43.90 s in that regime;
   run average 35.6 s. `T_G` = 0.2742 (derated) → `k_b*` **138 / 37.87 s** vs locked 160 /
   23.61 s — **reported, not applied**. Cause of the derate not established (DVFS or thermal
   trip; no readable GPU clock) and the one-closure/staging forms were not measured derated —
   that is the cheapest next measurement, `stream.py --chunks 10 --pub-every 160 --once-blocks 0`
   style re-run with the closure form reduced.
   **P3.6 DONE 2026-11-16 00:34 — Gate 4 PASS twice** → `docs/device/RESULTS_gate4_p36.md`,
   `tools/pipe/gate4.py`, captures `gate4_20261116_003217` / `…_003411`: PSS A+B peak
   **282.8 MB (0.27 GiB) vs the 2.2 GiB budget (8.2× headroom)**, `proc_a` holds one
   `rw-s` MAP_SHARED view with 10/10 imports, `proc_b` one fd and zero mappings. Two
   findings: **`smaps` under-reports a GPU process 2.93×** (the 164,192 kB `GL mtrack`
   row is invisible to `/proc/<pid>/smaps`) and the device had been carrying a **second
   46,399,488 B copy** of the block (`hv_block_int8.bin`, T_N-era) — removed with 128 MiB
   of other stale weight scratch, **137,502,720 B freed**; re-running `npu_block` needs
   its blobs re-pushed (no tool pushes them).
   **P3.7a DONE 2026-11-16 01:01 — Gate 3 measured and it FAILS** →
   `docs/device/RESULTS_gate3_parity.md`, `tools/pipe/gate3_parity.py`,
   `proc_b --dump-parity DIR --act payload|basis|window`. `proc_b` now dumps, per
   GEMM, the exact int8 activation each graph consumed and the int8 output it
   produced; the host recomputes `acc = Σ a·w` **exactly** (fp32 BLAS, K-split
   partials < 2^24, int64 accumulate; verified against brute-force int64
   including the all-−128 worst case) and applies NNAPI's requant
   `clamp(round(acc/127))`. Result: **25.8 % of each output tile is bit-exact**,
   and the correct region is structural — **the last 384-column block is exact
   for every row and row 0 is exact for every column** (`attn1.to_q` blocks
   1.1/1.1/1.1/**100 %**; rows 1…383 correct only in the last block). `ffn.up`
   is worse: 22 of 24 blocks at 1.4 %, then 45 %/92 %. The bit-exact regions are
   what prove the host reference, so this is the HAL, not the model. Ruled out by
   measurement: async write (150 ms re-dump identical), 4096-aligned output
   buffer (bit-identical), transpose, row stride, row/col permutation, n-shift,
   stale other-GEMM data, weight-byte echo. The dense payload saturates 45 % of
   outputs, which is why `--act basis` exists (every output then equals a stored
   weight byte). No lock touched, no rate edited: `T_N` was always declared
   latency-only.
   **P3.7b DONE 2026-11-16 01:25 — the bisect localizes it, and an exact recipe exists** →
   `docs/device/RESULTS_gate3_bisect.md`, tool `tools/pipe/gate3_bisect.py`,
   `proc_b --probe-dir DIR --probe-g G [--probe-plan "M,N,col0;…"]` (standalone probe mode, no ring,
   32 graphs over 5 captures). Measured, bit-exact = Δ 0 LSB vs exact integer accumulation:
   `N=384` **100.00 %** at `M=1/8/128/160/192/200`; `N=768` **100.00 %** at `M=128`; `N=1536`
   **100.00 %** at `M=1` only; `N=1536` at `M=256/320/384` → **50.67 / 50.60 / 25.83 %**. Same
   boundary on `ffn.up` (N=8960) ⇒ shape-driven, not weight-driven. **The 384-column graph at
   `M=384` does not compile at all** (`OP_FAILED` in 57–73 ms, three separate runs), so the
   "one graph per 384-column block" fix Gate 3 proposed is the exact graph this compiler refuses.
   The compiler hole is non-monotonic (`208×384` fails while `200×384` and `384×1536` compile) —
   recorded as a measured map, **no mechanism claimed**. Exact recipe = `N ∈ {128,384,768}` ×
   `M ≤ 200`: 4 column graphs × 2 row tiles for the eight 1536-wide GEMMs, 24 graphs
   (23×384 + tail 128 at `col0=8832`, also 100 %) for `ffn.up` ⇒ **112 executes/block vs 9 today**.
   Indicative (2 executes, ungated, *not* table-grade) warm exec 1.54–1.69 ms per `M=192×N=384`
   graph, 1.18–1.24 ms per `M=192×N=128`, 22.95 ms for one `M=192×N=8960` — extrapolated ≈0.18 s
   per exact block, *below* the 377.8 ms the current 9-graph stage costs.
   **P3.7c DONE 2026-11-16 01:36 — cost and exactness measured** → `docs/device/RESULTS_tn_recipe.md`,
   tool `tools/pipe/tn_recipe.py`, `proc_b --recipe 1 [--recipe-dump DIR]`, captures
   `tn_recipe_20261116_013019` … `_013625` (4 runs). One graph per column block at `M = 192` over
   1536 tokens = **56 compilations + 448 executes per block**: **0.7164 s/block**
   (715.8/717.2/717.8/716.4 ms, 0.3 % spread; pass 1 777–815 ms cold), **140 GOPS vs 265** for the
   9-graph stage ⇒ **1.90× its 377.8 ms**, and **448/448 dumped tiles bit-exact (Δ = 0 LSB)**
   including `ffn.up`'s tail-128 block. Compile **8.64 s per block** (154 ms/graph), separable
   from exec and amortisable if the graphs survive across chunks. `solve_cut.py` with `T_N` =
   0.7164 + 0.2580 downsep → **k_b\* = 166, period 24.47 s (+3.6 %)**, reported not applied.
   **P3.7d DONE 2026-11-16 02:09 — the exact recipe is co-run-priced and contention is a non-event**
   → `docs/device/RESULTS_corun_recipe.md`, tool `tools/corun/co_recipe.py` +
   `co_recipe_device.sh`, capture `corun_recipe_20261116_020926` (pilots `…_015944` / `…_020535`).
   `proc_b --recipe 1` and `gemm_bench block` in two processes, both stamping CLOCK_MONOTONIC:
   recipe **700.5 ms/block** contended vs **707.0 ms** in the passes before the GPU joined
   (−0.9 %), GPU **146.05 ms** contended vs **160.27 ms** solo, overlap **100.0 %**. `T_N` =
   0.7005 + 0.2580 downsep = **0.9585 s** ⇒ `k_b* = 166`, period **24.25 s (+2.7 %)** — reported,
   not applied. Required a **pass-major** rewrite of `run_recipe` (all 56 graphs compiled up
   front, then each pass walks the block) so that a pass is contiguous in time and can be
   classified solo vs contended. **NEXT ACTION: (a) re-run `python tools/pipe/tn_recipe.py --dump`
 to re-establish 448/448 exactness under the pass-major binary; (b) the exact stage's
 `ffn.down` under contention (the 0.2580 s downsep is still solo); (c) then the real
 Phase-3 remainder: `--decode 1` is still a no-op stub, Gate 1's PSNR/SSIM leg and
 `calibrate.py` are the host-only work, and Gate 5 is unrun.**
 **P3.7e DONE 2026-11-16 02:31 — (a) closed, and it moved the number** →
 `docs/device/RESULTS_tn_recipe.md` §P3.7e, capture `tn_recipe_20261116_023152` (gate PASS,
 TPU 27.0 vs 27.2): **448/448 tiles bit-exact (Δ = 0 LSB)** under the pass-major binary, and the
 block costs **646.69 ms** steady (646.4/646.7/646.7/647.0, stdev **0.21 ms = 0.03 %**), pass 1
 668.5 ms, compile 9.63 s, **155 GOPS**, **1.71×** the 377.8 ms 9-graph stage — i.e. pass-major is
 **9.7 % cheaper** than P3.7c's gemm-major 716.4 ms, not merely classifiable. **The confound:
 P3.7d's 707.0 "solo" / 700.5 "contended" both sit ~9 % above a floor-started solo run and bracket
 it, so "contention is a non-event (−0.9 %)" is not established** — candidates are a TPU derate
 above ~34 °C (the co_run reached 45 °C; the 9-GEMM soak saw +0.35 % only over 25→31 °C) or
 order/state (the co_run's 4 pre-GPU passes read 708–720 at a *cooler* start). `T_N` = 0.6467 +
 0.2580 downsep = **0.9047 s ⇒ `k_b*` = 165, 24.32 s**; the contended candidate (0.9591 s) gives
 166 / 24.47 s — the **cut is insensitive (165 vs 166)**, both reported, lock untouched.
 Two host defects fixed en route: `run(["bash", …])` resolved to **WSL bash** (subprocess searches
 System32 before PATH ⇒ the build silently never ran, rc 127), now routed through
 `probe_host.bash_bin()` for every tool; and a single confirm sample no longer aborts a floor-grade
 run (it re-enters the wait loop).
 **P3.7f DONE 2026-11-16 02:38 — the solo rate is settled and does not derate in 26→33 °C** →
 `RESULTS_tn_recipe.md` §P3.7f, capture `tn_recipe_20261116_023810` (40 passes, gate PASS at
 TPU 26.0): **647.93 ms** steady median (n=39, stdev 1.76 ms, 643.8–651.7 = 1.23 % spread),
 first-10 vs last-10 **+0.16 %**, slope **+0.0375 ms/pass** while the TPU rose **26 → 33 °C**
 (battery flat 22.7). Two floor-started runs now agree to **0.2 %** (646.7 / 647.9). This leaves
 the P3.7e confound asymmetric: **order/state is supported** (a clean start lands at 647–648 every
 time, never 707) and **thermal above ~34 °C is untested** — 40 passes only reach 33 °C, whereas
 the co_run climbed to 45 °C on the GPU's heat. The decisive run therefore has to walk the
 temperature range *and* bank floor-grade solo passes in one go.
   **P3.7g DONE 2026-11-16 02:40 — the co-run settles contention, and kills the thermal story ** →
   `RESULTS_corun_recipe.md` §P3.7g, capture `corun_recipe_20261116_024049` (`--delay 6 --reps 24`,
   gate PASS at TPU 26.0, overlap **99.06 %**, TPU **26 → 45 °C**): contended recipe
   **635.8 ms/block** (per-pass warm median 637.3, n=24, 634.5–659.4) vs **646.7 / 647.9 ms solo** from
   two GPU-free runs, GPU block **158.95 → 146.08 ms**. **The delay is measured from launch, so the
   compile lands inside it and every pass here was contended** (`npu_solo_*` null for that reason) —
   which makes it sharp: **no contention penalty** (contended ≈ solo, if anything slightly faster) and
   **no derate at 45 °C** (post-join passes flat 633.6–639.0 while the zone climbed). That kills
   P3.7d's 707.0 baseline *and* the "derate above 34 °C" candidate. **Still unexplained: the 11 %
   between the 02:09 run (707) and tonight (637) on the same binary/block; the only environmental
   change is that the device network/background services were ON at 02:09 and OFF since.** If
   background CPU load inflates the 448 synchronous executes, the network-on rates in
   `RESULTS_stream_p35.md` (32.72 s period) carry ~10 % too — hence next action (a).
   Host-only work that can run in parallel: ~~fold `T_V` = 10.640 (hybrid) into `configs/rates_preset_a.json`~~
   — **done 2026-11-15 21:3x** (k_b* re-solved: 160, period 23.61 s).
 **P3.7h DONE 2026-11-16 03:00 — (a) closed: the network costs nothing, and the "11 %" is a *reboot*
 → `RESULTS_tn_recipe.md` §P3.7h, capture `tn_recipe_20261116_025927` (Wi-Fi associated **and
 validated**, 2/2 ping 0 % loss, gate PASS at TPU 25.0): **639.44 ms** steady (n = 7, stdev 2.82) —
 1.1 % *cheaper* than P3.7e's 646.69 and 1.3 % cheaper than P3.7f's 647.93, both network-**off**. So
 background connectivity does not inflate the 448 executes and the **32.72 s stream period carries no
 network term**. `T_N` = 0.6394 + 0.2580 downsep = **0.8974 s** ⇒ k_b* = 165, 24.32 s.
 `/proc/uptime` (2,852.67 s at 03:10) puts the boot at **02:22:27**, and every exact-recipe capture
 sorts perfectly by it — **pre-boot 700–717 ms** (716.4 / 710.3 / 707.0 / 700.5), **post-boot
 636–651 ms** (646.7 / 647.9 / 635.8 / 639.4) ⇒ the 11 % is **boot-to-boot device state**, not
 contention, not heat, not load. Consequence: **P3.7e's "pass-major is 9.7 % cheaper" is retracted
 as an order effect** (it compared a post-boot number with a pre-boot one).**
 **P3.7i DONE 03:14 — same-boot order A/B: the order costs nothing** → new flag
 `proc_b --recipe-order pass|gemm` (both orders hold all 56 graphs; only the loop nesting differs) +
 `tn_recipe.py --order`; captures `…_031159` (gemm) / `…_031414` (pass): **649.52 ms** vs
 **651.23 ms** at starts 1 °C apart — 0.26 %, inside the 1.5 % within-run spread. Same five post-boot
 points also give the recipe a **start**-temperature term: 639.4 @ TPU 25.0, 647.9 @ 26.0, 646.7 @ 27.0,
 651.2 @ 31.0, 649.5 @ 32.0 = **1.8 % across 7 °C (+0.15 %/°C)**, versus **+0.16 % within** a run
 (P3.7f) — the start sets the regime, the run does not drift. k_b* = 165 at both ends (0.8974 / 0.9095
 s), lock untouched. **Quote a block time with its start temperature.**
 **NEXT ACTION: (a) `ffn.down` under contention** — **DONE 03:47, see P3.7j below**; (b) then the
 real Phase-3 remainder: `--decode 1` is still a no-op stub, Gate 1's PSNR/SSIM leg and
 `calibrate.py` are host-only work, and Gate 5 (the ≥10-chunk end-to-end run) is unrun.**
 **P3.7j DONE 03:47 — `T_N`'s last solo term is priced, and it is not contended either** →
 `RESULTS_corun_recipe.md` §P3.7j, tool `tools/corun/co_downsep.py` + `co_downsep_device.sh`,
 capture `corun_downsep_20261116_034737` (gate PASS TPU 27.0 vs 27.0, battery 24.0; same boot).
 Four arms in one `adb shell`: `npu_block downsep` (7×FC`[1536,1280]`+6×ADD, the 0.2580 s term)
 measures **62.769 ms/tile solo (n=48) ⇒ 0.2511 s/block**, **63.334 ms/tile contended (n=119) ⇒
 0.2533 s/block = +0.90 %**, and the co-run's own non-overlapped tiles read 63.132 — the delta is
 inside the arm-to-arm spread, so **no contention penalty on the down projection**. Same-session
 controls: recipe **637.13 ms** (n=7, 0.4 % from P3.7h ⇒ same boot), GPU **160.79 → 146.70 ms**
 contended, GPU-side overlap **99.9 %**. **`T_N` exact is now co-run end to end: 0.6371 + 0.2533 =
 0.8905 s/block** (0.8974 with the solo downsep, −0.8 %) ⇒ `k_b*` **165, period 24.21 s, 7.06×**
 vs locked 160 — reported, not applied, and insensitive (165 either way). Two harness defects
 fixed: **`npu_block downsep` was unrunnable from CLI defaults** (`G_PC` defaults to 1 = per-column,
 which `FULLY_CONNECTED` rejects at `addOperation` → `BAD_DATA` in 150 ms; the mode now forces
 per-tensor like `attn`), and **`$10` in POSIX `sh` is `${1}0`** (turned `--recipe-reps 8` into
 100). New: per-tile `TS|downsep` stamps and a `--from-logs TS` re-parse path (raw logs +
 `.ctx.json` sidecar written *before* analysis, so an analysis bug costs a re-parse, not a run).
 **NEXT ACTION: the real Phase-3 remainder, in this order — (a) `--decode 1` is still a no-op stub**
 (it prints the 10.640 s hybrid reference and executes nothing): **its input now exists, see P3.8a
 below — the execution itself (`npu_block convseq`) is the next device run**; (b) Gate 1's
 FP32-vs-INT8 PSNR ≥ 28 dB / SSIM ≥ 0.92 leg and `calibrate.py` remain the only host-only
 work and `.venv-ref` exists; (c) Gate 5, the ≥10-chunk end-to-end run with the decode in it.**
 **P3.8a DONE 2026-11-16 04:12 — the VAE decoder's INT8 artifact, i.e. the bytes `--decode 1` will
 execute** → `docs/device/RESULTS_vae_int8.md`, tool `phase1_quant/vae_int8.py`, artifacts
 `artifacts/vae/{weights_int8.bin,scales_f32.bin,bias_f32.bin,vae_int8.json}` (2.7 s to regenerate).
 36 decoder layers, **100 temporal taps, 592 `CONV_2D` executes/chunk == the plan's count**;
 **72,685,152 B INT8 unpadded** (= `configs/vae/layers.json` exactly) → **72,751,648 B** with
 4096-B tap windows (+0.091 %), **42,636 B** of FP32 per-output-channel scales, **42,636 B** of FP32
 biases. Layout **OHWI per tap** (transpose of `[Cout,Cin,kT,kH,kW]`) and proved, not asserted: a
 random activation through the checkpoint weight vs the dequantised INT8 taps under the same 2-D
 SAME conv gives **0.93 / 1.47 / 1.40 % relFro, cosine 0.99996 / 0.99989 / 0.99990** on `conv_in`
 and the first two resnet convs. Verifier **PASS: 100/100 taps byte-identical** to a from-source
 re-quantisation. **Two findings: (1) the locked per-output-column scheme, applied over the whole
 layer, costs `up_blocks.{0,1}.upsamplers.0.time_conv` their Gate-1 cosine — 0.997397 / 0.997484,
 2 of 36 layers — while per-(channel, tap) scales bring the worst layer in the decoder to 0.999910
 (measured side by side, not adopted); (2) bias stays FP32 and is flagged `quantised: false`,
 because an INT32 bias is in `SA*SW` units and `SA` is `calibrate.py`'s output — nothing invented an
 activation scale.** Mean relFro **2.000 %**. Nothing locked was touched.
 **P3.8b DONE 2026-11-16 04:37 — the decode EXECUTES, and `CONV_2D` takes per-channel scales** →
 `docs/device/RESULTS_vae_convseq.md`, tool `tools/npu/vae_seq.py` + `npu_block convseq`, capture
 `vae_seq_smoke_20261116_043719`. The artifact's **36/36 layers, 100 graphs, 592/592 executes**
 (= the plan's count) ran on one file-mapped 72,751,648 B copy; **9,065.3 ms of compute** vs the
 plan's 10,554.4 per-op sum (**0.859×**, 1,129 GOPS), compile 186 ms/graph, process wall 43.18 s.
 **The fork is answered: `CONV_2D` ACCEPTS `TENSOR_QUANT8_SYMM_PER_CHANNEL`** (add 0, finish 0,
 delegated 1, compile 85.8 ms) — 36/36 layers compiled per-channel, 0 fallbacks — so the FC's
 per-tensor restriction is an op hole, not the device, and the two `time_conv` cosine 0.9974
 layers are a live cost of the locked scheme. Per layer the agreement is *not* uniform: latent
 34×60 layers run **1.31–1.46× the per-op sum** (fixed ~0.9 ms/dispatch) while 272×480 run
 **0.86×** — quote a per-layer decode time with the tool that produced it. **13.7 s of the 22.78 s
 wall is this harness's own epilogue** (int32 tap accumulate + scalar double requant), so T_V only
 holds if that is fused/replaced. Thermal **started 0.9 °C over the gate** (TPU 25.0 vs 24.1,
 cold battery) and ended 41.0 → mechanism-grade, stated. Not wired: residual/shortcut adds,
 SiLU, temporal-shuffle offset, and `SA` is the declared nominal 1/127 → **no quality claim.**
 The wiring is derived+asserted for all 36 layers (3→5→9 frames, 34×60→272×480): the shape chain
 closes. Trap: manifest scale offsets are **BYTES** (÷4), caught by an assertion.
 **NEXT ACTION (device, one bounded run): the cool re-run for a table-grade start** —
 **DONE 05:0x, see P3.8c below**.
 **P3.8c DONE 2026-11-16 04:58:56 — the decode is priced at a table-grade start, and its `CONV_2D`
 class has a temperature term the FC stage does not** → `docs/device/RESULTS_vae_convseq.md` §P3.8c,
 capture `vae_seq_cool_20261116_045856`. **Start peak 24.0 °C against a `battery+3` of 24.1 °C**
 (battery cold at 21.1) ⇒ table-grade on the strict zero-slack rule, the 0.9 °C P3.8b missed.
 **592/592 executes, 8,834.7 ms of compute** (−2.5 % vs P3.8b's 9,065.3) = **0.837× the plan's
 10,554.4 ms** per-op sum, **579 GMAC/s = 1,158 GOPS** at 2 ops/MAC (the P3.8b table's 1,129 is the
 same number in that convention — the artifact's `gops` field is MAC/s, hence an apparent 2×).
 Walk 24.365 s, compile 18.64 s (186 ms/graph), and **the harness's own epilogue is still 15.53 s =
 64 % of the walk**. Per-plane exec/plan: 1.285 / 0.957 / 0.774 / 0.825 (34×60 → 272×480), agreeing
 with P3.8b within 6 % per plane ⇒ the per-layer structure is the shapes', not the thermal start.
 **NEW: four identical-shape layer pairs, early vs late in one walk (TPU 24 → 41 °C) read 0.992 /
 1.015 / 1.018 / 1.052** — median **+1.7 %**, worst **+5.2 %** on the largest plane 15.3 s apart,
 a monotonic rise the FC stage does not show (+0.35 % over 24.6 s) ⇒ **quote a decode time with its
 start temperature**. Cut consequence, reported not applied: `T_V` = 8.8347 + 0.0855 (Mali attn)
 = **8.920 s** re-solves to `k_b*` **163 / 24.06 s** vs 165 / 24.32 s on the plan's `T_V` — a 16 %
 cheaper decode moves the cut 2 blocks and 1.1 %, so the cut is insensitive to it; lock untouched.
 Defect fixed: `cool_rerun.py` returned 3 on a single failed confirm sample (it threw away 17 min of
 cooling at 04:39 on a 1 °C-quantised TPU zone) and now re-enters the wait loop, as P3.7e claimed.
  **NEXT ACTION (host-only, needs no phone): `calibrate.py` — deliverable 4.** Gate 1's PSNR/SSIM
 leg is now **done and PASSing** (`phase1_quant/psnr_gate1.py`, 40.22 dB / 0.9708 weight-only), so
 the remaining host-only blocker is the activation scale: `calibrate.py` must run the FP32 forward
 and record `max|x|/127` per (step, block, GEMM) — it is the only source of `SA`, and `SA` is what
 turns "the decode executes" into "the decode decodes" (and what makes the device's INT8 activations
 comparable to the 40.22 dB weight-only number). Each decoder pass is 30.5 s on `.venv-ref`, so the
 whole 36-layer sweep is minutes. Then wire `proc_b --decode 1` to call `convseq` instead of printing
 the reference, then Gate 5 (≥10 chunks with the decode in it).**
 **P3.9 DONE 2026-11-16 05:49 — `calibrate.py` exists, so `SA` is measured and the INT8
 activation path has a number** → `phase1_quant/calibrate.py`, `docs/device/RESULTS_calibrate.md`
 (scales in the gitignored `artifacts/calib/act_scales_{vae,dit}_20261116_0540*.json`), and
 `docs/device/RESULTS_gate1_actquant.md` (capture `gate1_psnr_20261116_054851`). Host-only, FP32
 forwards in `.venv-ref`. **DiT: 1800 scales** = 6 locked timesteps × 30 blocks × 10 base GEMMs at
 M = 1530 with **real umT5-XXL** conditioning (bf16, 512 tokens, 87 s; 15.3 s per step). SA
 0.002466 … 0.286219; **1261/1800 instances exceed the nominal `1/127`** (`attn2.to_q` 18.8×,
 `ffn.net.2` 5.5×) and 539 sit below it (`attn2.to_out` 0.50×), so no single activation scale is
 even directionally right. Two consistency proofs fell out rather than being asserted:
 `attn2.to_k` == `attn2.to_v` == **0.005011 in all 180 instances** (one RMS-normalised cross-attn
 context) and `attn1.to_q/_k/_v` agree exactly (one shared input). Hidden-state max|x| peaks at
 **359.04 vs the fp16 ceiling 65504 ⇒ 182× headroom**, so the lock's "fp16 residual + per-GEMM
 calibrated input scale" needs no narrower term. A *static* per-GEMM max costs up to **7.65×** of
 resolution, which is why the table is per (step, block, GEMM). **VAE: 36/36 layers** measured on
 the very tensors the artifact's taps consume (max kept across the decode's frame chunks, 2–3
 calls/layer). SA 0.004822 … 0.252453 ⇒ **34 of 36 layers need a LARGER scale than the nominal
 `1/127`** (median 4.94×, worst **32.06×** on `up_blocks.1.resnets.0.conv_shortcut`), 2 need
 smaller — i.e. the P3.8b/P3.8c `convseq` walks were clipping on 34 of 36 layers, which is the
 measured reason they "execute but do not decode". **Consequence for Gate 1:**
 `psnr_gate1.py --act-scales` now runs the decode with **both operands INT8** →
 **38.73 dB / SSIM 0.9672 PASS** vs weight-only **40.22 dB / 0.9708** (reproduced to 2 dp ⇒ the leg
 is deterministic): the activation operand costs **1.48 dB / 0.0036 SSIM**, concentrated on the
 frames with the most headroom (f0 −9.47 dB, steady-state f4–f8 −0.34…−1.23 dB). No device rate
 touched; `T_N`/`T_G`/`T_V`/`k_b*` unchanged. **NEXT ACTION: (a) wire `SA` into the device — **DONE 06:10, see P3.10 below**; (b) `proc_b --decode 1` calls `convseq` instead of printing 10.640 s; (c) Gate 5, ≥10 chunks with the decode in it.**
 **P3.10 DONE 2026-11-16 06:10 — the per-layer `SA` is wired into the device, and it costs no latency**
 → `docs/device/RESULTS_vae_convseq.md` §P3.10, capture `vae_seq_sa_20261116_061040`, tools
 `phase1_quant/vae_sa.py` + `npu_block convseq`. The artifact is now
 **`artifacts/vae/act_scales_f32.bin` = 36 × f32 = 144 B**, derived from the newest vae calibration
 and verified float-for-float against the source (delta 0), in `vae_int8.json`'s layer order.
 `convseq` reads `/data/local/tmp/vae_sa.f32` and applies per layer: input AND output operand scale
 = `SA[i]` (was 1/127 for all 36), bias as **real INT32 in `SA·SW` units per output channel** (was
 zeros in per-channel mode; 0 int32 clamps), and the output requant `g = SA[i]/SA[i+1]` — to the
 **consumer** layer, so the byte-copy wiring stays a copy, replacing the old `g = SW[c]/SA` that
 multiplied an activation accumulator by a weight scale. Measured: 36/36 layers, 100 graphs,
 **592/592 executes**, `sa=table` accepted on all 36, `pc=1` 0 fallbacks, compile 18.74 s, wall
 43.30 s, start TPU 24.0 vs gate 23.6 (**0.4 °C over — mechanism-grade, stated**). **Sum of the 592
 execute walls 8,822.6 ms vs P3.8c's 8,834.7 at 1/127 ⇒ the SA wiring costs −0.14 %, i.e. nothing**,
 and the table is verified on the device (36 `sa=` values equal the bytes to 2.8e-07 abs = print
 precision; `sa_out[i] == SA[i+1]` for all 35 links). The epilogue is now instrumented:
 **12,027.9 ms = 52.9 % of the 22,730.5 ms walk** + 1,879.9 ms accumulate/input assembly = 61.2 %
 harness overhead, so P3.8c's "15.53 s epilogue" was epilogue **+ accumulate** and the epilogue
 itself is 3.50 s cheaper now. Saturation is counted, not hidden: **17/36 layers saturate nothing,
 median 0.63 %, worst 38.57 %** (`up_blocks.3.resnets.2.conv2`); `corr(log10(SA_in/SA_out), clamp) =
 0.50` ⇒ the scale ratio is **not** the driver, the missing residual/SiLU wiring and an unmeasured
 per-layer **output** range are. `T_V` unchanged (8.8226 + 0.0855 = 8.908 s, `k_b*` = 163), no lock
 touched, **no quality claim: it executes in the right numeric range, it still does not decode.**
 **NEXT ACTION: (a) `calibrate.py` must record the per-layer OUTPUT range (its `max_abs_out` reads a
 flat 1.0, so it was never a real output calibration) → table becomes 2 × 36 floats, and wire the
 residual/SiLU/clamp path — that is what turns "executes" into "decodes"** — **DONE host-side
 06:37, see P3.11 below**; (b) `proc_b --decode 1` calls `convseq` instead of printing 10.640 s;
 (c) Gate 5, ≥10 chunks with the decode in it.**
 **P3.11 DONE 2026-11-16 06:37 — the wiring is measured, and it is a NORM, not a rescale** →
 `docs/device/RESULTS_vae_wiring.md`, tools `phase1_quant/vae_wiring.py` +
 `phase1_quant/check_norm_path.py`, artifacts `artifacts/vae/{norms_f32.bin (31,104 B, 29 gammas,
 7,776 floats),vae_wiring.json (128 ops),act_scales2_f32.bin (288 B = 2×36 f32)}`. `calibrate.py`'s
 output range is now a **forward** hook (`max_abs_out` was the whole decode's output = 1.0 for all
 36 rows) and `resnets[].max_abs_out` is the **post-add** tensor, which no conv owns. Measured:
 **SA_in 0.004822…0.252453, SA_out 0.011217…0.431392, SA_out/SA_in 0.275…15.505 (median 1.305)**;
 post-add range = **1.03×…4.76×** conv2's own (median 1.97×) ⇒ a residual add at conv2's scale
 clips. The 89× step between `mid_block.resnets.0.conv1` (54.79) and `.conv2` (0.612) is
 `WanRMS_norm` = `F.normalize(dim=1)*sqrt(C)*gamma` + SiLU: **hand-applied it gives 0.6124 vs the
 measured 0.6124, rel 0.00 %** (PASS) — and two cross-checks agree for free (the post-add range of
 `up_blocks.{0,1}.resnets.2` equals the following `time_conv`'s input range exactly; conv-less edges
 like `time_conv→resample.1` agree to 0.98/1.00 while norm-bearing ones span 0.17–89). Plan:
 36 conv + 29 norm + 29 SiLU + 14 save + 14 add + 2 temporal_shuffle + 3 upsample2d + 1 attn.
 **Ordering defect recorded:** the artifact indexes `conv1,conv2,conv_shortcut` but `forward` applies
 the shortcut FIRST ⇒ the SA table's index order is a storage order, not an execution order (3
 positions differ, all `decoder.up_blocks.1.resnets.0`); every conv op now carries `artifact_index`.
 **Not wired into `convseq` yet** — norm/SiLU/shuffle/upsample/add are unexecuted, so the decode
 still executes without decoding, no quality claim, and no device rate changed.
 **NEXT ACTION (device, one bounded unit): implement the P3.11 plan in `convseq`** — **DONE 07:02, see
 P3.12 above**: implemented as `npu_block convwire` over a 101-op program, not inside `convseq` (so the
 P3.8–P3.10 captures keep their meaning). 592/592, saturation 0.0721 % worst. **Then `OP DUMP` — DONE
 07:36, see P3.13 above**: the walk emits its final codes + the latent it consumed; the device decode
 scores **11.60 dB / cosine 0.219** against fp32 from the same latent, the tap→frame labels were
 **mirrored (FIXED)** and the `CONV_2D` HAL is **faithful at 98.59 % bit-exact**. **New NEXT ACTION, in
 order: (a) host-side program simulator over `vae_wire.txt` + per-layer device probes
 (`--dump-layer L`) to localize which GLUE op is wrong — the convs and the HAL are exonerated;
 (b) `post_quant_conv` is missing from the program (report, don't fold); (c) per-tap bias should be
 given to tap 0 only (recovers ~1 %/layer); then `proc_b --decode 1` → `convwire`, then Gate 5.**
 **P3.14 DONE 08:16 — (a) is closed and it found a real defect, worth +8.84 dB** →
 `docs/device/RESULTS_vae_glue.md`, tool `tools/npu/vae_sim.py`, device run
 `vae_wire_fix_20261116_080721`, analysis `vae_psnr_fix_20261116_080912`. The temporal
 upsample gather was wrong (`(t/2, t%2)` instead of "frame 0 = `b[0]` half 0 only, then
 both halves of each later buffer frame"); the frame count is `2T-1` either way, so it was
 silent. Device decode vs fp32: **11.60 dB / 0.2190 → 20.44 dB / 0.8967**. Layers 0–10
 track the reference at 0.965–0.996, layer 21 (the second upsample) at 0.9488 ⇒ no second
 structural defect; the rest is accumulating scheme cost.
 **NEXT ACTION, in order: (a) score the three accumulating-scheme candidates with the
simulator, no device needed — **DONE 08:33, see P3.15 at the top**: two are worth
−0.29 dB and +0.04 dB (the scheme is *not* the residual error) and the missing
`post_quant_conv` is worth **+7.32 dB** — then **layer 34 is the named defect site**;
(b) the layer-11/20 3-frame-vs-2-frame `time_conv` support; (c) then `proc_b --decode 1`
→ `convwire`, and Gate 5.**
10. Phase 4 tables, report.
11. **Layer 34's remaining question — the oracle-input probe (P3.16).** Both static
   candidates are priced and refuted (`RESULTS_vae_l34.md`: `add34` +0.0016 dB, `coutpc`
   +0.0015 dB, layer 34 stays 0.7149 in every arm). What is left is whether conv2 adds its
   own error or inherits conv1's (input already at 0.8618): run each conv on **the
   reference's own input** quantised at the program's `SA_in` and score its output —
   intrinsic-bad ⇒ the site (weights/scale/gather), input-inherited ⇒ upstream. Host-only,
   one walk (~3.5 min), extends `vae_sim.py` with `--oracle`; the reference trace already
   dumps a conv input for the two `resample.1` layers, so this generalises an existing hook.

## Decisions already locked (do not reopen without saying why)

- Backbone: **DCM-Wan2.1-1.3B**, `N = 6`, `L = 30`, `B = 180`; expert switch `κ = 3`.
- Quantisation: **per-output-column symmetric INT8**, one FP32 scale per column, base weights only;
  bias/norm/AdaLN/embeddings stay FP16.
- Detail expert stored **factorised** (rank-128 `A`,`B` + its own timestep block), never merged.
  ⚠️ measured cost of FP16 factors: **+226 ms per detail block** vs **+41 ms** if the factors were
  INT8 — a decision to put to the student, not to change quietly (`RESULTS_gemm_mali.md` §6).
- One **block-level** cut, one handoff per chunk, `k_b` from co-run rates with `T_V`.
- Layout: **row-major `[N_out, K]` INT8** — verified to be exactly what both `char4`+`arm_dot`
  and NNAPI `FULLY_CONNECTED` want.
- The NPU stage is a **NNAPI graph**, not custom kernels.
- Attention **de-fused** (QK → softmax → AV), 256-row score block.
- No fabricated device numbers: unknown ⇒ `null` + "TBD (blocked: …)".

## Kernel facts worth not rediscovering

- `clFinish` is **not** a completion barrier on Mali r54p2 — close every timing with a blocking read.
- Global size is `ceil(N/BN)·(BN/TN)`; the `round_up` form corrupts memory and surfaces only as
  `CL_EXEC_STATUS_ERROR (-14)`.
- **32 int accumulators maximum** per thread; 64 spills (112–144 B/thread) and costs 4.5×.
- Predicate the epilogue — a `break` in an unrolled loop spills the accumulators.
- `arm_dot_acc` is a real SDOT; measured ceiling **1.77 TOPS**.
- **The tiled GEMM needs N and M on a 64-multiple.** At N=6120 the tail output tile writes past `C`;
  the only symptom is `CL_EXEC_STATUS_ERROR (-14)` on the completion read, while the timing prints a
  physically impossible 10,502 GOPS (measured 2026-11-15, `attn_mali_*`). Pad the shape and record the pad.

NNAPI / Edge TPU (paid for 2026-11-15, evidence in `RESULTS_npu_tn.md`):

- **A model needs `identifyInputsAndOutputs`.** Operands + `addOperation` alone compile to
  `model_finish → BAD_DATA`. That was the first hour of the `T_N` harness.
- **`getSupportedOperationsForDevices` requires a FINISHED model** — called before
  `model_finish` it returns `BAD_STATE` and tells you nothing.
- **A 10-op chained block graph reboots the phone** inside `Compilation_finish` (uptime resets).
  It validates and reports full support first, so validation is not a viability test.
- **`ffn.down` (K=8960) never finishes compiling** → `DEAD_OBJECT` after ~62 s, reproducible fresh.
- **`FULLY_CONNECTED` rejects `TENSOR_QUANT8_SYMM_PER_CHANNEL`** (`addOperation` → `BAD_DATA(4)`).
- **`CONV_2D` (VAE decode) compiles only at N=1 above ~2.4 MB/frame**: N=3 at 0.78 MB/frame
  compiles in 141.6 ms; N=3/N=5 at 1.4 MB and N=3/N=9 at 6.3 MB fail after 31–36 s, while the
  same 37.6 MB tensor compiles in 263 ms at N=1. Support says `delegated=1` either way.
  `CONV_2D` also reaches **685–1842 GOPS** vs 285 GOPS for FC on the same silicon.
- `getDuration(DURATION_ON_HARDWARE)` reports ~6 µs for a 0.9 GMAC GEMM → unusable; time the
  synchronous `execute`.
- **A bare `"bash"` from Python is WSL's bash, not git-bash** (measured 2026-11-16). `subprocess`
  searches `System32` before `PATH`, so `run(["bash", "C:/…/build.sh"])` returns rc 127
  (`/bin/bash: C:/…: No such file or directory`) and the build **silently never runs** — WSL mounts
  this repo at `/mnt/c`, so neither the `C:/` nor the `/c/` form resolves. Always go through
  `probe_host.bash_bin()` / `run()`; never spawn a bare `bash` for a build script.
- **`$10` in a POSIX `sh` device script is `${1}0`** (measured 2026-11-16). `RECIPE=$10` after a
  `$1` of 10 produced **`--recipe-reps 100`** — 74 s of device time instead of 15, visible only in
  the `reps=` field of the SUMMARY line. Use `${10}` for every positional past the ninth.
- **`npu_block`'s `G_PC` defaults to 1 (per-column), and `FULLY_CONNECTED` rejects per-channel at
  `addOperation`** — so any mode that does not force `G_PC = 0` dies in ~150 ms with
  `FATAL|addOperation … rc=4(BAD_DATA)` and never reaches the compiler. `attn` already did this;
  `downsep` did not, and was **unrunnable from the CLI defaults** until 2026-11-16 (the 0.2580 s
  reference must have passed the per-tensor argument explicitly). Every new FC mode needs it.
- **`smaps` does not see a GPU process's memory.** `/proc/<pid>/smaps_rollup` reported
  `Pss` 85,137 kB for `proc_a` while `dumpsys meminfo <pid>` reported `TOTAL PSS`
  **249,290 kB** for the same PID — the gap is the **`GL mtrack`** row (164,192 kB), the
  Mali driver's accounting of the CL buffers the process owns, which never appears in
  smaps. **Ratio 2.93×; always quote `dumpsys` for the Mali stage.** The control: `proc_b`
  has no `GL mtrack` and its two sources agree to 0.1 % (measured 2026-11-16,
  `RESULTS_gate4_p36.md`).
- **`ls -l /proc/<pid>/fd` counts descriptors, not stored copies.** `proc_b` showed 1 fd on
  the artifact steady-state and 4 transiently while the 9 graphs compiled; what matters is
  that every one resolves to the same path and that only one file of that size exists on
  the device (a leftover T_N-era `hv_block_int8.bin` broke the one-copy claim until it was
  deleted).
- **A causal conv3d's weight slice 0 is applied to `x[f-2]`, not `x[f]`** (measured
  2026-11-16). `WanCausalConv3d` pads two zeros at the FRONT of the time axis, so the tap
  decomposition of a `kT=3` conv is `tap t ↔ frame f-(kT-1-t)`. `vae_export` labelled slice 0
  `"f"`, and every wire program built from that label read the wrong frames for 36 layers —
  it executes, and its saturation looks *better*, while the decode is uncorrelated with the
  reference (cosine 0.056 vs the checkpoint; 1.00005 when mirrored). Any tap→frame table
  must be checked against the module, not derived from the shape.
- **`CONV_2D` on this TPU is numerically faithful — `FULLY_CONNECTED` is not** (measured
  2026-11-16, `RESULTS_vae_psnr.md`): one layer's device codes are **98.59 % bit-identical**
  to an exact emulation of the program (100 % within ±1, cosine 1.000312), whereas the FC
  hole (Gate 3) returned 25.8 % for wide shapes. So a decode error is a *program* bug or the
  glue, and the conv must be exonerated before anything else is blamed.
- **`vae.decode` applies `post_quant_conv` (16→16, non-identity) before `decoder.conv_in`**,
  and that op is not in the artifact — a device-vs-host comparison that forgets it is
  comparing two different inputs (it produced a 0.46-cosine false alarm on the first attempt).
- **`--convs N` + dump is post-NORM, not post-conv**: the walk breaks at the *next* conv, after
  the intervening ops have run. A per-layer parity probe must dump inside the conv.
Host-side model traps (paid for 2026-11-16, `phase1_quant/calibrate.py`):

- **The DCM checkpoint is not a stock diffusers state dict.** Every attention/ffn projection is
  peft-wrapped: `blocks.N.attn1.to_q.base_layer.weight` (480 such tensors) plus 480
  `lora_A`/`lora_B` expert factors. `from_pretrained` cannot load it; build from
  `WanTransformer3DModel.from_config`, strip `.base_layer`, drop the LoRA keys, and **assert
  `missing == 0`** — that is what proves the mapping. The only unexpected tensors are the 12
  `condition_embedder_lora.*` (DCM's text-embedder LoRA, deliberately not applied).
- **`scheduler.set_timesteps(n)` is not the DCM schedule.** For the locked 6-step run it returns
  `[999, 937, 857, 750, 600, 376]`, not `[1000, 992, 982, 949, 905, 810]`; its *sigma* convention
  is the plain rectified-flow identity `sigma = t/1000` (measured to 1e-3). Use the locked
  timesteps and derive sigma from the identity.
- **umT5-XXL is 22.7 GB fp32 on disk.** Load it bf16 (11 GB) **before** the DiT and free it —
  28 s to load, 87 s for a 512-token encode on 6 CPU threads; co-resident with the 5.7 GiB
  transformer it is an avoidable 17 GB peak.
- **`AutoencoderKLWan.decode` is iterative** — it walks the latent with cached conv state, so one
  decoder layer fires **2–3 times per decode**, at 1-frame shapes. A calibration hook that
  *assigns* keeps only the last chunk (caught in the first attempt); it must accumulate the max.
  Built-in invariance check: `attn2.to_k`/`to_v` read the same RMS-normalised cross-attn context,
  so their SA is identical to the last digit in all 180 (step, block) instances.
- **A `FULLY_CONNECTED` output is only partly the GEMM.** Measured 2026-11-16
  (`RESULTS_gate3_parity.md`): **25.8 %** of each `[384, N]` output tile is bit-exact
  against exact integer accumulation — the **last 384-column block** for every row and
  **row 0** for every column — and the rest is data that is not any FC result over these
  weights (not another GEMM's, not weight bytes). For `ffn.up` (N=8960) the tail blocks are
  45 %/92 % rather than 100 %. Not an async write (150 ms re-dump identical), not buffer
  alignment (4096-aligned output → bit-identical), not transpose/stride/permutation/shift.
  **Consequence: measure latency with this stage, never numerical correctness.**
  `--act basis` (`A[r][r]=127` ⇒ every output equals a stored weight byte) is the probe
  that makes the check information-rich; the dense residual saturates 45 % of outputs at
  `SA = 1/127` and hides the failure in rounding noise.
- **`FULLY_CONNECTED` has a shape hole, and the exact shapes are small** (measured 2026-11-16,
  `RESULTS_gate3_bisect.md`, 32 probe graphs on two GEMMs). Bit-exact (Δ 0 LSB):
  `N = 384` × `M ≤ 200`, `N = 768` × `M = 128`, `N = 1536` × `M = 1`. Refused at compile
  (`OP_FAILED` in 57–73 ms, not a timeout): `N = 384` × `M ≥ 208`, `N = 768` × `M ≥ 192`,
  `N = 1536` × `M ≤ 192` — **non-monotonic**: `200×384` compiles, `208×384` does not,
  `384×1536` does, `192×768` does not. Wide graphs at `M = 256…384` compile and run but are only
  25.8–50.7 % exact, and the surviving region is **not** a fixed rule: at `M=384` the last
  384-column block is 100 % while at `M=256/320` all four blocks are ~50.7 %. So there is no
  "correct subset" to exploit — either use the small exact shapes or accept a wrong result.
- **All 56 exact-recipe graphs can be held at once** (measured 2026-11-16, `RESULTS_corun_recipe.md`):
  compiling every column block up front and then running *pass-major* keeps each pass contiguous
  in time, which is what lets a pass be classified as "ran alone" vs "ran while the GPU was busy".
  P3.7c freed each GEMM's slots before compiling the next, which forced gemm-major order and made
  every pass span the whole run — the same numbers (710.3 vs 716.4 ms solo) but unclassifiable.
- **Device rates are bounded by the boot and by the start temperature** (measured 2026-11-16,
  `RESULTS_tn_recipe.md` §P3.7h/§P3.7i). `/proc/uptime` showed a reboot at 02:22:27 that splits every
  exact-recipe capture cleanly: **pre-boot 700–717 ms, post-boot 636–651 ms** for the same binary,
  block and geometry. So (a) quote the boot / session with every rate, and (b) never compare a number
  taken before a reboot with one taken after — that is what broke P3.7e's "pass-major is 9.7 %
  cheaper" (the real order effect is **0.26 %**, measured same-boot with
  `proc_b --recipe-order pass|gemm`). Within one boot the block also tracks its **start** temperature
  (+0.15 %/°C over a 25→32 °C start) even though a running pass barely drifts (+0.16 % over 26→33 °C).
- **A load experiment must be started AFTER the thermal gate opens, inside the same device command
  as the measurement.** Four `yes` loops at 108 % each (464 % of 800 % CPU) launched *before* the gate
  drove TPU 25 → **56 °C** and skin2 to 43.5 in under a minute; the gate then refused for 420 s and no
  artifact was written (measured 2026-11-16). Gate → load → measure, in one `adb shell`.

## Definition of done for the course project

1. Artifact + quantiser + parity report (Gate 1) committed.
2. Device probe + measured rates + solved cut (Gate 2) committed. ← partially: GEMM done
3. Two-process pipeline streaming ≥10 chunks on the phone without manual intervention (Gates 3–5).
4. `results/tables.md`: configurations, cut ablation, memory, quality (PSNR/SSIM/LPIPS), SLO sweep,
   plus a prediction-vs-measured table from Phase 2.
5. Report-ready list of deviations from the paper, each with the measurement that forced it.
