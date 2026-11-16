# P3.8b — the VAE decode executes: 592 `CONV_2D` dispatches on the real INT8 artifact

Tool: `tools/npu/vae_seq.py` → flat plan `docs/device/vae_seq.txt` → device mode
`npu_block convseq <max_layers> <pc>` (`tools/android/npublock/npu_block.cpp`).
Capture `vae_seq_smoke_20261116_043719` (the name says smoke; the arg-order defect below made it
run **all 36 layers**, so it is the full decode). Binary `bf40ad1b…`, weights
`9b925a049d58932f`.

`--decode 1` was a no-op stub because no INT8 VAE weights existed. `phase1_quant/vae_int8.py`
(P3.8a) wrote them; this run is that stub's execution.

## The fork: `CONV_2D` **accepts** per-channel scales

`FULLY_CONNECTED` rejects `TENSOR_QUANT8_SYMM_PER_CHANNEL` at `addOperation`
(`BAD_DATA(4)`), which is why the DiT stage is per-tensor. On the same HAL,
`CONV_2D` takes it:

```
PROBE|conv2d_per_channel|add_rc=0(NO_ERROR)|finish_rc=0(NO_ERROR)|support_rc=0|
delegated=1|compile_rc=0(NO_ERROR)|compile_ms=85.8
```

and then **36/36 layers compiled per-channel, 0 fallbacks** (`pc_fallbacks=0`,
`tap_failures=0`). So the decode can carry the locked per-output-column scales; the NPU's
per-tensor restriction is an op-specific hole, not a device-wide one. Consequence for the fork in
`RESULTS_vae_int8.md` §1: the two `time_conv` layers' cosine 0.9974 is a **live** cost of the locked
scheme (per-(channel,tap) scales reach 0.999910), not something the HAL forces.

## Measured

| quantity | value |
|---|---|
| layers executed | **36 / 36** (the plan's whole decoder) |
| taps / graphs compiled | **100**, one `CONV_2D` graph per (layer, tap) |
| executes | **592 / 592** — equal to `plan_npu.json`'s `total_executes_per_chunk` |
| compute time, sum of per-dispatch walls | **9,065.3 ms** |
| plan reference (per-op sum, `T_V`) | 10,554.4 ms → **convseq is 0.859× it** |
| aggregate rate | **1,129 GOPS** (plan 969.7) |
| MACs | 5,117,308,600,320 (5117.3 GMAC, matches the plan) |
| compile | **18.56 s** total, **186 ms/graph** |
| process wall | 43.18 s |
| weights | ONE file-mapped copy, 72,751,648 B, windowed per tap (`createFromFd` + `setOperandValueFromMemory`) |

Thermal: start TPU **25.0** / battery 21.1 / skin 23.4 / VIRTUAL-SKIN 22.7 → gate
`battery+3 = 24.1` ⇒ **started 0.9 °C over the gate** (cold battery makes the gate strict, and
the TPU's idle floor is 25.0). End TPU **41.0**. Per the project rule this is **mechanism-grade,
not table-grade** — stated, not dropped. The plan's 10.554 s remains the table entry.

## The aggregate agrees, the per-layer numbers do not

`exec/plan` per layer is not one number, it is a **shape trend**:

| shape class | layers | per-execute | exec/plan |
|---|---|---|---|
| latent 34×60, 384 ch | 11 | 3.6–4.1 ms | **1.31–1.46** (up to 1.78 on `conv_in`) |
| 68×120 | 9 | 5.7–10.9 ms | 0.75–1.02 |
| 136×240 | 7 | 13.9–24.1 ms | 0.74–0.78 |
| 272×480, 96 ch | 8 | 25.8–26.2 ms | **0.86–0.87** |
| `conv_out` | 1 | 15.5 ms | 0.97 |

Small shapes run **~40 % slower** than their per-op sweep rate and the big ones **14–26 %
faster**, so the per-op sum is not a per-layer predictor even though the total lands 14 % low.
Candidates, not established: a fixed per-dispatch cost (~0.9 ms on the latent shapes) that the
sweep's back-to-back `tiles` amortised, versus a different weight hand-off and a hotter device for
the large shapes. **Quote a per-layer decode time with the tool that produced it.**

## Two costs this measured that are not hardware

1. **The host epilogue.** Summed over layers the walls are **22.78 s** while the dispatches are
   9.07 s, so **13.7 s** went into this harness's own per-tap work: an int32 accumulate over 592
   output planes plus a scalar `double` dequant→requant per element. That is a naive epilogue
   (integers + NEON would be an order of magnitude cheaper, and a fused NNAPI epilogue removes it
   entirely) — but it is the honest cost of *this* execution, and it is why "T_V ≈ 9–10.6 s" only
   holds if the tap accumulation is not done this way.
2. **Compile dominates a first chunk.** 100 graphs at 186 ms = 18.56 s per chunk unless the
   compilations are held across chunks (the pass-major trick `proc_b` already uses).

## What is deliberately not wired (no quality claim)

Residual adds, the resnet-shortcut addition (`up_blocks.1.resnets.0.conv_shortcut` executes on a
**zero** activation, its shape still asserted against the layer that produced it), the SiLU/clamp
epilogues, and the exact temporal-shuffle offset. Activation scale is the **declared** nominal
`SA = 1/127` because `calibrate.py` does not exist yet, so activations saturate and the outputs are
not the decoder's. This run measures execution, dispatch count and latency.

## Plan validation that fell out of it

The wiring is *derived and asserted*, not assumed: every layer's `[frames, cout, h, w]` must be
consumed by the next layer's `[frames, cin, h, w]` under one of five declared rules (copy;
temporal unshuffle 2C→C over 2t−1 frames; that + 2× nearest; 2× nearest only; zero-fed shortcut).
All 36 layers resolve and the chain closes — **3 → 5 → 9 frames** through the two `time_conv`
stages and **34×60 → 68×120 → 136×240 → 272×480** spatially, ending at 9 × 3 × 272 × 480. A
manifest that disagreed with itself would have been a hard error, not a silently wrong execute.
Trap paid for: `vae_int8.json` stores scale offsets in **BYTES** (÷4 for the element index) — the
same confusion that produced 72 spurious failures in the P3.8a verifier, caught here by an
assertion before any device time was spent.

## Defect paid for (host side)

`vae_seq.py` passed `convseq <layers> <pc>` but the binary read `max_layers`/`pc` from `argv[4]`/
`argv[5]`, so `--layers 2` ran all 36. Harmless here (the full run is the deliverable) but it makes
a prefix run silently a full run; arg positions now agree.

## Next

1. **Cool re-run for a table-grade start** (`cool_rerun.py --cmd "python tools/npu/vae_seq.py
   --run --out vae_seq_cool"`): the mechanism-grade run started 0.9 °C over the gate.
2. `--decode 1` in `proc_b` can now call this same walk instead of printing the reference.
3. Parity still needs `SA` per layer from `calibrate.py` (FP32 forward through the decoder) plus
   the residual wiring — that is the difference between "executes" and "decodes".
