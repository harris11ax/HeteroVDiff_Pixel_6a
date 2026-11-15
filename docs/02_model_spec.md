# DCM-Wan2.1-1.3B — measured from the checkpoint

Everything below comes from `phase1_quant/inspect_weights.py` run against
`weights_raw/DCM_WAN/transformer/diffusion_pytorch_model.safetensors`
(HF `cszy98/DCM`, public). Reproduce with:

```
python phase1_quant/inspect_weights.py --manifest configs/block_manifest.json
```

## Headline numbers **[measured]**

| quantity | measured | blueprint / paper says |
|---|---|---|
| tensors | 1317, all **F32** (5.73 GiB on disk) | — |
| total parameters | **1,539,041,408** | 1.419 B semantic + 94.4 M LoRA + 25.6 M timestep = 1.539 B ✓ |
| — stock `diffusers.WanTransformer3DModel` from `config.json` | **1,418,996,800** | paper: semantic expert 1,419 M ✓ |
| — + LoRA factors | 94,371,840 | paper: 94.4 M ✓ |
| — + detail-expert timestep block | 25,672,768 | paper: 25.6 M ✓ |
| sum check | 1,418,996,800 + 94,371,840 + 25,672,768 = **1,539,041,408** | ✅ exact |
| params per DiT block | **49,586,432** (uniform across all 30) | — |
| INT8 region per block | **47.29 MiB** (30 regions = 1,418.7 MiB) | paper 47.3 MiB ✓ |
| base attention (INT8) | 566,231,040 params = **0.527 GiB** | paper 541.4 MiB ✓ |
| base FFN (INT8) | 825,753,600 params = **0.769 GiB** | paper 787.5 MiB ✓ |
| LoRA A + B | 94,371,840 params = **90.0 MiB INT8 / 180.0 MiB FP16** | paper 94.4 M, 91.5 MiB ✓ |
| detail-expert timestep block | 25,672,768 params = **24.5 MiB INT8 / 49.0 MiB FP16** | paper 25.6 M ✓ |
| **everything in INT8** | **1,467.7 MiB** | blueprint: "~1.47 GiB" ✅ exact |
| **block regions only (INT8)** | **1,418.7 MiB** | paper's 1,423 MiB total ✅ (their +4 MiB = scales) |

The blueprint's 1.47 GiB and the paper's 1,423 MiB are **both right but count different things**:
1.47 GiB is every parameter at 1 byte; 1,423 MiB is base + scales + factors without the
top-level embeddings. The allocator only ever needs the **47.29 MiB per-block region**.

The per-block INT8 region (47.29 MiB) is the allocator unit — 30 regions, each far below any
plausible `AHardwareBuffer` ceiling. **No 32×32 HMX tiling is needed on Pixel** (see
`docs/03_spec_vs_reality.md` §4).

## One DiT block, exact tensor inventory (block 0; identical for all 30)

```
blocks.N.attn1.to_{q,k,v,out.0}.base_layer.weight   [1536,1536]   2.25 M each   self-attn
blocks.N.attn1.to_{q,k,v,out.0}.base_layer.bias     [1536]
blocks.N.attn1.norm_q / norm_k.weight               [1536]        RMSNorm across heads
blocks.N.attn2.to_{q,k,v,out.0}.base_layer.weight   [1536,1536]   2.25 M each   cross-attn
blocks.N.attn2.norm_q / norm_k.weight               [1536]
blocks.N.ffn.net.0.proj.weight                      [8960,1536]  13.13 M        up
blocks.N.ffn.net.0.proj.bias                        [8960]
blocks.N.ffn.net.2.weight                           [1536,8960]  13.13 M        down
blocks.N.ffn.net.2.bias                             [1536]
blocks.N.norm1.weight                               [1536]        LayerNorm, no affine
blocks.N.norm2.weight/bias                          [1536]        cross-attn pre-norm (affine)
blocks.N.norm3.weight                               [1536]
blocks.N.scale_shift_table                          [1,6,1536]    AdaLN: shift/scale/gate for MSA + FFN
blocks.N.attn{1,2}.{to_q,to_k,to_v,to_out.0}.lora_{A,B}.lora1.weight
                                                    [128,1536] / [1536,128]   rank-128 detail-expert LoRA
```

Per-block param split: 8 × 2.25 M (attention bases) + 2 × 13.13 M (FFN) = 44.25 M base, plus
4.7 M of LoRA factors per block (only used by blocks ≥ 0 when the detail expert is active —
the factors are stored for all 30 blocks).

Top-level (non-block) tensors that must also be quantised or kept in FP16:

```
patch_embedding.weight                [1536,16,1,2,2]   ← temporal patch = 1, spatial 2×2
proj_out.weight                       [64,1536]         ← 16ch × 2 × 2
condition_embedder.*                  25.77 M           ← semantic-expert timestep + text projection
condition_embedder_lora.*             25.67 M           ← DETAIL-expert timestep block (own copy)
norm_out.* / proj_out_lora.*                             ← detail-expert output head
```

## Architecture facts the kernels must honour (from `third_party/DCM/fastvideo/models/wan_hf/modeling_wan.py`)

| item | value | note |
|---|---|---|
| hidden dim `d` | 1536 | 12 heads × 128 |
| FFN | `Linear(1536→8960) → GELU(tanh≈) → Linear(8960→1536)` | ⚠️ **not SwiGLU** (blueprint §1 row 1 says SwiGLU) |
| attention | fused QKV **not** used: separate `to_q,to_k,to_v,to_out.0`, all with bias | |
| qk-norm | `rms_norm_across_heads` (RMSNorm over the full 1536, then reshape to heads) | must be inside the GEMM epilogue or a separate pass |
| RoPE | 3-axis, **adjacent-pair rotation**, `θ=1e4`, `max_seq_len=1024`; head_dim 128 splits **t=44, h=42, w=42** | applied to Q and K only, in FP64 in the reference impl |
| AdaLN | `(scale_shift_table + temb).chunk(6)` → shift/scale/gate for MSA and for FFN | `temb` is per-step, not per-block: compute once per step on CPU |
| residual stream | FP32 in the reference; paper keeps FP32 on NPU, FP16 between blocks on GPU | §4.4 |
| patchify | `[B,16,F,H/2,W/2] → [B, M, 64] → Linear(64→1536)` | `d_z = 16·2·2 = 64` |

## Expert routing (verified in code, not inferred)

`fastvideo/sample/inference_wan.py`: `if student and int(timestep[0]) <= 978: self.set_adapter('lora1')`
and the forward then uses `condition_embedder_lora`, `norm_out_lora`, `proj_out_lora`.
With `cus_timesteps=[1000, 992, 982, 949, 905, 810]` ⇒ **steps 0,1,2 = semantic**,
**steps 3,4,5 = detail** (`κ=3`, switch at block `κ·L = 90`).
Reference invocation (`third_party/DCM/scripts/inference/inference_wan.sh`):

```
--height 480 --width 832 --num_frames 81 --num_inference_steps 6 --guidance_scale 1
--embedded_cfg_scale 6 --flow_shift 7 --flow-reverse --seed 1024 --model_path ckpt/DCM_WAN/
```

⚠️ The checkpoint ships `scheduler/scheduler_config.json` = **UniPCMultistepScheduler
(flow_shift 3.0)**, but DCM inference **replaces** it with `InferencePCMFMScheduler`
(`fastvideo/distill/solver.py`) and the hardcoded 6 timesteps above. Use the scheduler from
`fastvideo/distill/solver.py`, not the one in the checkpoint, or you will not reproduce the
6-step trajectory.

## VAE (`weights_raw/DCM_WAN/vae/config.json`, `AutoencoderKLWan`)

```
z_dim = 16, base_dim = 96, dim_mult = [1,2,4,4], num_res_blocks = 2
temperal_downsample = [false, true, true]   ⇒ spatial 8×, temporal 4×
latents_mean / latents_std = 16-element vectors in the config   ← normalise BEFORE decode
```
Wan VAE temporal rule: output frames `= 4·F_z - 3` for causal decode with cache
(81 frames ⇔ 21 latent… with the shipped config 81 → 21 latents; **9 output frames ⇔ F_z = 3**).
The decoder is a stack of causal 3-D convs with a recurrent feature cache; decoding frame `0`
skips temporal upsampling. Paper decodes in **60×40 latent-pixel tiles, 3 per frame**, one tile's
cache resident.

## Resolution presets — computed from the real patch size

`M = F_z · (H_z/2) · (W_z/2)`, `F_z = (frames−1)/4 + 1`, `H_z = H/8`, `W_z = W/8`, patch `(1,2,2)`.

| preset | H×W | frames | F_z | H_z×W_z | **M** | full score matrix FP32 | 256-row score block FP16 | hidden state FP16 |
|---|---|---|---|---|---|---|---|---|
| Baseline | 256×256 | 9 | 3 | 32×32 | **768** | 27.0 MiB | 4.5 MiB | 2.25 MiB |
| Preset A | 480×272 | 9 | 3 | 34×60 | **1530** | 107.2 MiB | 9.0 MiB | 4.48 MiB |
| Preset B | 832×480 | 9 | 3 | 60×104 | **4680** | 1002.6 MiB | 27.4 MiB | 13.71 MiB |
| (paper) | 832×480 | 49 | 13 | 60×104 | 20280 | 18.8 GiB | 118.8 MiB | 59.41 MiB |

⇒ **de-fuse attention and tile the score matrix** (row blocks): at Preset B a materialised score
matrix is 1 GiB in FP32, which is why the paper's GPU path keeps a 256-row block (27.4 MiB here,
118.8 MiB at their `M`). This is not optional on a 6 GB phone.

## Cut-solver sensitivity (so you know what to expect before the phone arrives)

`k* = (180·T_N + T_V)/(T_N + T_G)`, valid `k_b ∈ [1,179]`; `k*` can land outside that range
when `T_V` is large relative to a block — then the integer search over `[1,179]` decides, and the
binding constraint is the NPU stage (`k_b = 179` gives the GPU 179 blocks and the NPU 1 + decode).

| `T_N` | `T_G` | `T_V` | `k*` |
|---|---|---|---|
| 0.10 s | 0.20 s | 10 s | 93 |
| 0.10 s | 0.20 s | 30 s | 160 |
| 0.20 s | 0.50 s | 30 s | 94 |
| 0.20 s | 1.00 s | 30 s | 55 |
| 0.50 s | 1.00 s | 60 s | 100 |
| 0.50 s | 0.50 s | 30 s | 120 |

The 9-frame workload makes `T_V` ~10–40 s instead of the paper's 267 s, so the balance point
sits **earlier** (smaller `k_b`) than the paper's 51 unless the NPU is very slow per block.
