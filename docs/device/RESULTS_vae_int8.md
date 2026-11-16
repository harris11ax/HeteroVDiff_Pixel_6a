# P3.8a — the VAE decoder's INT8 artifact (the bytes `--decode 1` will execute)

Tool: `phase1_quant/vae_int8.py` (`--verify` re-derives from the source checkpoint).
Inputs: `weights_raw/DCM_WAN/vae/diffusion_pytorch_model.safetensors` (507,591,892 B FP32, 194
tensors) + `configs/vae/plan_npu.json`. Outputs: `artifacts/vae/{weights_int8.bin, scales_f32.bin,
bias_f32.bin, vae_int8.json}` (gitignored, regenerable in **2.7 s**).

## Why this exists

`--decode 1` in the NPU stage is a no-op stub: it printed the `T_V` = 10.640 s hybrid reference and
executed nothing, because no INT8 VAE weights had ever been written — `npu_block conv` timed the
real decoder shapes against a deterministic LCG fill, which is a latency table and not an
execution. This artifact is the missing input.

## Measured

| quantity | value |
|---|---|
| decoder layers in the plan | **36** (35 causal 3-D convs + 1 mid-block 3-D attention) |
| temporal taps written | **100** (kT=3 for every conv except the 4-D resample convs) |
| `CONV_2D` executes/chunk the plan implies | **592** — artifact says 592, plan says 592 |
| INT8 weight bytes, unpadded | **72,685,152 B** = `configs/vae/layers.json` `weight_bytes_if_int8` total, exact |
| INT8 weight bytes with 4096-B tap windows | **72,751,648 B** (+**0.091 %**) |
| FP32 per-output-channel scales | **42,636 B** (10,659 scales = Σ`Cout`) |
| FP32 biases, unquantised | **42,636 B** |
| quantisation, mean relFro | **2.000 %** |
| quantisation, worst cosine | **0.997397** (`up_blocks.0.upsamplers.0.time_conv`) |
| layers below the Gate-1 bar 0.999 | **2 / 36** (both `time_conv`) |
| verifier | **100/100 taps byte-identical** to an independent from-source re-quantisation |

Layout: layer-major, tap-major inside a layer, and `[Cout, kH, kW, Cin]` (OHWI) inside a tap — the
transpose of the checkpoint's `[Cout, Cin, kT, kH, kW]`, which is the order `CONV_2D` reads. Each
tap starts on a 4096-byte boundary so it can be handed to `createFromFd` /
`setOperandValueFromMemory` as a window without a copy (the same trick `proc_b` uses for the DiT
block). The manifest carries **byte offsets**, not arithmetic, so the device indexes the file.

The layout is not merely self-consistent: `layout_probe` runs a random activation through both the
checkpoint weight and the dequantised INT8 taps under the same 2-D SAME conv and sums over taps —
`conv_in` **0.93 %** relFro / cosine **0.999957**, `mid_block.resnets.0.conv1` **1.47 %** /
**0.999891**, `conv2` **1.40 %** / **0.999902**. A transposed filter or a wrong tap order would show
up as a large error here, not as a plausible one.

## Two things this measured that the decisions did not expect

1. **The locked quantiser analogue costs the two `time_conv` layers their Gate-1 cosine.**
   Per-output-column scales computed over the layer's whole `[Cout, Cin*kT*kH*kW]` view (the direct
   analogue of the DiT's per-column scheme) give **0.997397** and **0.997484** on
   `up_blocks.{0,1}.upsamplers.0.time_conv` — below 0.999, on 2 of 36 layers. Measured side by side:
   **per-(channel, tap)** scales bring the worst layer in the whole decoder to **0.999910**.
   Reported, **not adopted** — the artifact keeps the locked scheme, and this is a fork for the
   student (the temporal upsample convs are 1.4× scale spread and are 2 of the 100 taps).
2. **Bias cannot be quantised yet, so it is stored FP32.** `CONV_2D` wants an INT32 bias in units of
   `SA*SW`, and `SA` (per-layer activation scale) is exactly what `calibrate.py` has to record — it
   does not exist. Writing `SA = 1/127` (what `npu_block conv` does for its LCG runs) would put a
   fabricated scale into a committed artifact, so the biases are recorded unquantised with
   `"quantised": false` in the manifest. A decode execution that uses them must dequantise to FP32
   and add on the host, or wait for `calibrate.py`.

## One-copy rule

Every tensor this artifact reads starts with `decoder.` and the verifier asserts it; the 300 DiT base
GEMMs in `artifacts/blocks/` are untouched. The VAE decoder is a different set of tensors, so this is
not a second INT8 copy of the base weights — the same reasoning that let the expert factors be FP16.

## Defect paid for (host-side)

The first verifier compared **byte** offsets from the manifest against **element** indices in the
freshly loaded scale/bias arrays. Layer 0 (offset 0) passed, layers 1…35 all "failed" — 72 spurious
failures that looked exactly like a write-path bug. Offsets in the manifest stay bytes (the device
wants bytes); the verifier now divides by the element size once, with a comment saying why.

## Next

`npu_block` gains a `convseq` mode that walks `vae_int8.json`: one `CONV_2D` per (layer, frame, tap),
int32 accumulation across taps, then the layer's residual/resample wiring — that is the execution
behind `--decode 1`. First question it must answer on the device: does this HAL accept
`TENSOR_QUANT8_SYMM_PER_CHANNEL` for `CONV_2D` (it is rejected for `FULLY_CONNECTED`), i.e. can the
decode use the per-column scales in `scales_f32.bin` at all, or is it per-tensor like the FC stage.
