# ⚠️ Blueprint vs. Tensor G1 reality

The blueprint was written by transplanting HeteroVDiff's **Qualcomm** mechanisms onto a
**Google Tensor G1**. Some of them do not port. This file lists each one with the evidence,
what to do instead, and which probe decides it. **Read this before writing Phase 3 code.**

## Summary table

| # | Mechanism | Blueprint assumes | Pixel 6a / Tensor G1 reality | Verdict | Fallback |
|---|---|---|---|---|---|
| 1 | NPU programming model | Hexagon + FastRPC + hand-written HVX/HMX kernels (`htp-ops-lib`) | Edge TPU reachable **only** through NNAPI (or a TFLite/LiteRT delegate); no public custom-kernel or FastRPC path | ❌ custom kernels impossible | build the suffix as a **NNAPI graph**; keep the same block-range interface |
| 2 | Attention inside the NPU graph | fused HMX FlashAttention | NNAPI has **no SDPA op** and **no 3D-RoPE op** | ⚠️ must decompose | QK `FULLY_CONNECTED` (batched) → `SOFTMAX` → AV `FULLY_CONNECTED`; RoPE as MUL/ADD on Q,K from precomputed tables |
| 3 | VAE decoder on NPU | INT8 QNN HTP graph, `Conv3d` in the vendor backend | NNAPI has **no `CONV_3D`** | ⚠️ must decompose or move | causal 3D conv → per-tap shifted 2D `CONV_2D` + `ADD`; or decode on Mali via OpenCL; or CPU |
| 4 | Weight layout | canonical **HMX 32×32 tile order** consumed by both engines | no HMX; NNAPI wants row-major `[N,K]`, which is *also* what Mali's `char4` dot wants | ✅ simpler than the paper | plain row-major `[N_out, K]` INT8 + per-column scales ⇒ one layout for both engines |
| 5 | Zero-copy handoff | `AHardwareBuffer_getNativeHandle()` → DMA-BUF fd → `SCM_RIGHTS` → `ANeuralNetworksMemory_createFromAHardwareBuffer` | `AHardwareBuffer_getNativeHandle` is **not in the NDK r27 public header** (dlsym it); NNAPI AHB memory exists (API 29); **CL side needs `cl_arm_import_memory`**, unverified on Mali-G78 | ⚠️ Gate 2 | **Plan B: file-backed `MAP_SHARED` mmap** — both processes map the same file ⇒ same page cache pages ⇒ one physical copy, no AHB, no SCM_RIGHTS; NNAPI via `ANeuralNetworksMemory_createFromFd` (API 30), OpenCL via `CL_MEM_USE_HOST_PTR` |
| 6 | Workload | VFI: 9-frame chunk, 1 conditioning start frame + 8 synthesised | checkpoint is **text-to-video only**: `image_embedder: null`, `added_kv_proj_dim: null` ⇒ **no image conditioning exists** | ❌ not possible with these weights | reframe as *9-frame T2V chunk stream* (see §5) |
| 7 | Engine speed ratio | NPU ~3.4× faster per block than GPU ⇒ NPU takes the suffix | unknown; Mali-G78 MP20 INT8 vs Tensor Edge TPU is a different comparison entirely | **[UNVERIFIED]** | measure `T_G`, `T_N`; if the GPU is slower per block than the NPU, the cut moves or the roles swap (§6) |
| 8 | `arm_dot` usage | `sum = arm_dot(a, b, sum)` | `arm_dot` is **2-argument**; the accumulating form is `arm_dot_acc(a, b, acc)` under `cl_arm_integer_dot_product_accumulate_int8` | ❌ blueprint kernel does not compile | use the correct builtins (probe tries all three paths) |
| 9 | FFN | SwiGLU | GELU(tanh≈) two-layer FFN, no gate | ❌ doc bug | one GEMM pair, no gate multiply |
| 10 | Thermal gate | cold start ≤ 30 °C ambient, pens, USB-C hub | G1 idles near 30 °C in a 24 °C room; the gate may be unreachable | ⚠️ soften | gate on `≤ ambient + 3 °C`, log ambient, note drift |

## 1. How the NPU is reachable at all — and why the paper's kernels do not port

The paper's NPU executor is a **custom FastRPC package** that calls `htp-ops-lib` HMX/HVX
kernels directly, bypassing the QAIRT graph runtime, because "the QNN HTP backend prevents
external operators from using HMX" (paper §5.1). It also patches two library bugs (a
lookup-table softmax exp accurate only for small scores, and a mis-ordered FP32→FP16 cast).

Pixel 6a has **no equivalent**: the Tensor Edge TPU is not publicly programmable. Google's own
issue trackers confirm there is no TPU delegate for LiteRT on Tensor phones, and that LiteRT in
Play Services ships only **GPU + XNNPACK** delegates (`google-ai-edge/LiteRT#969`, `#6501`).
NNAPI is **deprecated as of Android 15** (`developer.android.com/ndk/guides/neuralnetworks/
migration-guide`) though the driver is still present and callable — the NDK still ships
`android/neuralnetworks.h` and `libneuralnetworks.so` (verified in NDK 27.3.13750724).

**Consequence:** your "Process B" cannot contain hand-written kernels. It is a graph builder.
Design it as an object with the paper's interface — `run_blocks(start_block, count, hidden_state,
latent, timestep)` — so the coordinator code is identical to the paper's, and the graph builder
is swappable for XNNPACK/CPU if the TPU path fails Gate 0.

Analogue worth knowing: the paper's *reason* for bypassing the vendor runtime (custom kernels,
no HMX for external ops) evaporates on Pixel — you get the vendor runtime and nothing else. That
is a **weaker** position: expect per-block latency to be dominated by NNAPI graph dispatch
unless you compile one graph per block-range and reuse it (do that).

## 2. NNAPI op coverage — what a Wan block needs, what NNAPI has

| Wan block operation | NNAPI primitive | status |
|---|---|---|
| LayerNorm / RMSNorm (incl. `rms_norm_across_heads`) | `MEAN`, `RSQRT`-free: use `MUL`/`ADD`/`REDUCE`… no `RSQRT` op | ⚠️ decompose (or fold into GEMM epilogue on the GPU side) |
| AdaLN modulation `x*(1+scale)+shift` | `MUL`, `ADD` (broadcast) | ✅ |
| Linear 1536→1536 / 1536→8960 / 8960→1536 | `FULLY_CONNECTED` (rank-2/3 tensors OK) | ✅ |
| Per-column INT8 weight scales | `ANeuralNetworksModel_setOperandSymmPerChannelQuantParams`, channelDim=0 | ✅ (feature level 3, API 30 — verify on device) |
| GELU(tanh≈) | `LOGISTIC`/`TANH`/`MUL`/`ADD` | ⚠️ decompose |
| QK^T, AV | `FULLY_CONNECTED` on 3-D tensors (batch × M × K) | ✅ but no fused softmax |
| Row softmax | `SOFTMAX` (beta, axis) | ✅ |
| 3-D RoPE adjacent-pair rotation | `MUL`, `ADD` with precomputed cos/sin tables | ✅ but tables are `[M × 128]` per shape — precompute offline |
| Cross-attention to umT5 embeddings | `FULLY_CONNECTED` with constant K/V | ✅ |
| Causal 3-D conv (VAE) | **no `CONV_3D`** | ❌ decompose into shifted `CONV_2D` taps |

Quantisation model mismatch to watch: NNAPI's signed int8 is `ANEURALNETWORKS_TENSOR_QUANT8_SYMM`,
not a `TENSOR_INT8` type — the probe initially failed to compile on exactly this. Per-tensor
scales are `scale`/`zeroPoint` in `ANeuralNetworksOperandType`; per-channel needs the symm-per-
channel call above.

## 3. Zero-copy: three routes, in the order you should try them

**Route A — matches the paper (AHB + FD passing).**
1. `AHardwareBuffer_allocate()` `AHARDWAREBUFFER_FORMAT_BLOB`, usage `CPU_READ|CPU_WRITE|GPU_DATA_BUFFER`.
2. DMA-BUF fd: `AHardwareBuffer_getNativeHandle()` — **not declared in NDK r27's
   `android/hardware_buffer.h`** (verified: the header exposes only allocate/acquire/release/
   describe/lock/sendHandleToUnixSocket/recvHandleFromUnixSocket/isSupported/getId).
   Resolve it with `dlsym(RTLD_DEFAULT, "AHardwareBuffer_getNativeHandle")` and read the fd from
   `native_handle_t->data[0]` (the blueprint's `ANativeWindow_Buffer*` cast is wrong).
3. Send the fd with `SCM_RIGHTS` (`sendmsg`/`recvmsg`) — the blueprint's helper code is fine.
4. NPU side: `ANeuralNetworksMemory_createFromAHardwareBuffer()` (API 29+).
5. GPU side: `clImportMemoryARM(ctx, flags, NULL, (void*)(uintptr_t)fd, size, &err)` —
   requires `cl_arm_import_memory` + `CL_IMPORT_TYPE_DMA_BUF_ARM`. **Unverified on Mali-G78.**

Route A dies on step 5 or 4 on many devices. The probe (`ZEROCOPY_*` keys) tells you which.

**Route B — file-backed mmap, one page cache, no IPC** *(recommended default if A dies)*
Both processes `mmap(MAP_SHARED)` the same artifact file. Linux gives one set of physical pages;
each engine's runtime maps it. NPU: `ANeuralNetworksMemory_createFromFd(fd, size, offset, prot)`
(API 30, works with any file/shmem region). GPU: `clCreateBuffer(CL_MEM_USE_HOST_PTR, ptr…)`,
which on a UMA Mali driver is a no-op mapping (verify with a write-back test). This keeps the
*contribution* — one stored copy, two engines, no per-engine conversion — while dropping the
paper's Qualcomm-specific IPC. Say so explicitly in the report; it is a documented deviation,
not an accident.

**Route C — last resort:** per-engine private copy delivered by `read()` (2× memory) — this
**fails Gate 2** and must be labelled as such in the write-up, with the measured memory cost.

## 4. Weight layout: the paper's canonical format is unnecessary here

The paper stores INT8 in HMX 32×32 tile order because the Hexagon matrix unit requires it in
hardware, and pays for it with a 4-term byte-address formula (Eq. 2) so the GPU can still form
`char4` operands. Pixel has no HMX. `FULLY_CONNECTED` in NNAPI takes `[N_out, K]` row-major with
K contiguous, and `char4` operands are exactly 4 consecutive K values of one output column — the
same bytes. ⇒ **store plain row-major `[N_out, K]` INT8 + one FP32 scale per output column**, and
both engines read it natively. Simpler than the paper; document the simplification.

## 5. The workload does not exist as described

Blueprint §1 rows 3 & 6: "9-frame latent video chunks (**1 conditioning start frame + 8
synthesized** intermediate/end frames)" — i.e. frame interpolation. The DCM-Wan2.1 checkpoint is
**text-to-video**: `transformer/config.json` has `image_dim: null`, `added_kv_proj_dim: null`, and
the model file has no image-embedding tensors (verified: `image_embedder = None` in
`modeling_wan.py`). There is **no mechanism to condition on a start frame** with these weights.

Verified latent arithmetic still holds for 9 frames: `F_z = (9−1)/4 + 1 = 3`, so a 9-frame clip is
`3 × H_z × W_z` tokens — that part of the blueprint is right.

Options, in order of preference:
1. **Reframe as a 9-frame T2V chunk stream** (recommended): same `B=180`, same cut, same pipeline;
   the "frame interpolation" framing is dropped. Nothing in HeteroVDiff depends on VFI.
2. Keep VFI semantics with **overlapping chunks** (generate 9 frames, slide by 4, cross-fade the
   seam) — still T2V, but the output reads as interpolation. Cheap to implement; say it is a
   presentation trick.
3. Switch model to Wan2.1-I2V-14B — 14 B params, does not fit the INT8 budget on a 6 GB phone
   (3.7 GiB base + activations). Not viable.

## 6. The rate regime may invert the design premise

The paper's cut assumes the NPU is ~3.4× faster per block than the GPU (3,352 ms vs 12,633 ms on
Adreno 830 vs Hexagon V79). Nobody knows the Mali-G78 MP20 vs Tensor Edge TPU ratio. Three
outcomes and what each means:

| measured | consequence |
|---|---|
| `T_G < T_N` (GPU faster per block) | the cut formula still works; GPU simply takes the larger prefix; report the ratio — it is a *stronger* result than the paper's |
| `T_G ≈ T_N` | few blocks left for a pipeline; gain comes mostly from overlapping the VAE decode; report honestly |
| `T_G ≫ T_N` (GPU too slow, like the paper) | the pipeline still wins on *stream* throughput as long as `k_b·T_G < (B−k_b)·T_N + T_V`; but a tiny `k_b` means the GPU contributes ~nothing — then state that HeteroVDiff's premise (GPU as a real second stage) does not hold on Tensor G1 and report the measured ceiling |

If the GPU turns out unusable, the *design-preserving* fallback is to make the second stage the
**CPU cluster** (2× Cortex-X1 + 2× A76 via XNNPACK INT8) instead of the Mali GPU. Same
architecture — two processes, one INT8 artifact, one block-level cut, co-run rate measurement —
and it is defensible in the report as "the second engine that Tensor G1 actually exposes".

Do **not** pre-commit to any of these. Gate 0 measures `T_G/T_N/T_V` first.

## 7. Concrete blueprint bugs (fix before use)

| where | problem | fix |
|---|---|---|
| §5.1 OpenCL kernel | `arm_dot(a_val, b_val, sum)` — `arm_dot` takes **2** args | `sum += arm_dot(a,b);` or `sum = arm_dot_acc(a,b,sum);` |
| §5.1 | `#pragma OPENCL EXTENSION cl_arm_integer_dot_product : enable` is right, but the *extension strings* to test are `cl_arm_integer_dot_product_int8` / `..._accumulate_int8` / `..._accumulate_int16` / `..._accumulate_saturate_int8` | check those strings, not the bare name |
| §5.2 | `AHardwareBuffer_getNativeHandle(ahb)` cast to `ANativeWindow_Buffer*`; returns `const native_handle_t*`, and is absent from the NDK header | `dlsym` + read `data[0]` |
| §1 row 1 | "SwiGLU FFN" | GELU(tanh≈) FFN |
| §1 row 3/6 | "frame interpolation, 1 conditioning frame" | T2V only (§5 above) |
| §1 row 4 | "Edge TPU via Android NNAPI" with "custom OpenCL INT8 GEMM on Mali" — correct in outline | but the NPU side is a **graph**, not kernels |
| §5.3 Gate 1 | "combined RSS < 2.2 GiB" | RSS on Android includes shared pages; measure `dumpsys meminfo` **PSS** and report both, or count tensor bytes like the paper does |
| §1 row 2 | "single canonical INT8 base-weight artifact (~1.47 GiB across all 30 blocks)" | correct at 1 byte/param: measured **1,467.7 MiB** for all 1,539,041,408 params; 30 block regions alone are 1,418.7 MiB |

## 8. Thermal realism

Pixel 6a on Tensor G1 throttles early and its skin sensor sits far from the SoC. `dumpsys
thermalservice` reports several zones (`BIG`, `MID`, `GPU`, `SKIN`, `BATTERY`, …) with different
idle temperatures; "≤ 30 °C" may be unattainable if the room is warm, and it is not the right
gate anyway — what matters is **reproducible starting state**. Gate on:
`max(SoC zones) ≤ ambient + 3 °C`, ambient measured with a sensor or the battery zone, and log
both. `tools/probe/probe_host.py --cooldown 30` implements the polling.

## 9. What ports unchanged (do not second-guess these)

- The flattened `B = 180` trajectory and single block-level cut — `N`, `L` do not care about the SoC.
- Per-output-column INT8 with `s_j = max|W_·j|/127` — one scale vector per GEMM.
- Factorised detail expert (base `W` + rank-128 `A`,`B`, delta applied at full precision) — the
  paper's argument (adapter magnitude ≈ ⅓ of requantisation noise) applies identically.
- Co-run measurement of `T_G, T_N, T_V`, then `period(k_b)` evaluation over integer neighbours.
- Two-process isolation, one handoff per chunk, eventfd-free double buffering (a second
  handoff slot is enough — see `docs/05_phase_plans.md`).
- GPU-side FP16 hidden state + per-GEMM input scale calibration; NPU-side FP32 residual stream.
- De-fused attention (QK → softmax → AV) with a row-blocked score buffer.
- Report shape: NPU-only / GPU-only / DP / Hetero-Tensor / HeteroVDiff, SLO sweep, memory table.
