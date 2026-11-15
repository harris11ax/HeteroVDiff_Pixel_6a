# Gate 0 — measured device profile (Pixel 6a / Tensor G1)

Raw captures: `docs/device/probe_20261115_135227.{json,md}` plus the probe2 runs logged below.
Probe sources: `tools/android/probe/probe.cpp` (capabilities), `probe2.cpp` (NNAPI + zero-copy).
Driver: `tools/probe/probe_host.py`.

## Device

| | |
|---|---|
| model / device | Pixel 6a (`bluejay`), serial 25311JEGR11811 |
| SoC | `gs101`, `ro.soc.manufacturer=Google`, `ro.soc.model=Tensor` (Tensor G1) |
| Android | **17** (`ro.build.version.release`), SDK **37**, vendor API level 32 |
| RAM | `MemTotal 5,723,472 kB` = **5.46 GiB** (blueprint assumed "~4.2 GiB usable") |
| RAM free at probe time | MemAvailable 2.17 GiB |
| NNAPI runtime | `hal_neuralnetworks_darwinn` running, `nnapi_native.current_feature_level=7` |

## Mali-G78 (Process A candidate) — **viable, with a caveat**

| property | value |
|---|---|
| name / version | `Mali-G78 r0p0`, OpenCL 3.0 `v1.r54p2-01eac0` |
| compute units | 20 |
| global mem | 5,589.3 MiB |
| `CL_DEVICE_MAX_MEM_ALLOC_SIZE` | **4,095.5 MiB** (Adreno 830 advertises 1024 MiB) |
| `cl_arm_integer_dot_product_int8` | **present** |
| `cl_arm_integer_dot_product_accumulate_int8` | **present** |
| `cl_arm_integer_dot_product_accumulate_saturate_int8` | **present** |
| `cl_khr_integer_dot_product` | present (also the CL 3.0 core-style extension) |
| `cl_arm_import_memory` (+ `_dma_buf`, `_host`, `_android_hardware_buffer`) | **present** |
| `cl_khr_external_memory_dma_buf` | present |

**First INT8 GEMM measured** (`arm_dot`, one thread per output, no tiling — deliberately naive):
`128 × 8960 × 1536` in **338.7 ms = 10.4 GOPS**. That is ~1–2 % of what Valhall should reach, so
kernel quality is the whole ballgame on this engine: a tiled/blocked GEMM is required before any
`T_G` claim is meaningful. Treat 10.4 GOPS as the floor, not the rate.

## AHardwareBuffer ceiling — **1792 MiB** (paper's Snapdragon died at 2 GiB)

Sweep at `AHARDWAREBUFFER_FORMAT_BLOB`, usage CPU_READ|CPU_WRITE|GPU_DATA_BUFFER:

| size | isSupported | allocate | dma-buf fd |
|---|---|---|---|
| 16 … 1792 MiB (every step tried) | 1 | **OK** | fd 11, `/proc/self/fdinfo` size = requested + 4 KiB |
| **2048 MiB** | 1 | **FAIL** | — |

⇒ one per-DiT-block region (47.29 MiB) is trivial; even the whole 1.4 GiB artifact fits in a single
buffer. The paper's per-block split was forced by a 511 MiB limit — **that constraint does not
exist here**, so per-block regions are a *design choice* (finer handoff/remap granularity, and they
keep each engine's mapped footprint small), not a necessity. Say that explicitly in the report.

## Zero-copy routes — see the measured section below

Both backings are accepted by NNAPI (`createFromAHardwareBuffer` and `createFromFd` return 0) and both
import into OpenCL. The full verification (GPU write visible through the host pointer) is in
**"Zero-copy — Gate 2 passes on both routes"** below.

## NNAPI — the Edge TPU *is* reachable, but it has a hard token ceiling

```
NNAPI_DEVICE_COUNT_rc=0 count=2
NNAPI_DEV[0].NAME=google-edgetpu     TYPE=4 (ACCELERATOR) fl=1.3
NNAPI_DEV[1].NAME=nnapi-reference    TYPE=2 (CPU)          fl=1.3
```
`google-edgetpu` **reports support** for `FULLY_CONNECTED` (`getSupportedOperationsForDevices` → true)
and **executes** INT8 FCs from an `AHardwareBuffer`-backed weight operand.

### Measured: one INT8 FC, K=1536, N=8960 (the FFN-up GEMM), isolated process per shape

| M (tokens) | compile | wall (3 iters) | effective |
|---|---|---|---|
| 32 | OK | 6.25 ms | 141 GOPS (overhead-dominated) |
| 128 | OK | **10.72 ms** | **329 GOPS** |
| 256 | OK | **24.80 ms** | **284 GOPS** |
| 384 | OK | **37.43 ms** | **282 GOPS** |
| **448** | **OP_FAILED** | — | ceiling hit |
| 512 … 1530 | **DEAD_OBJECT** | — | HAL dead after the first refusal |
| N=1536 @ M=1530 (fresh process) | OP_FAILED | — | same token limit, independent of N |

⇒ **the per-graph activation ceiling is ~384 tokens** for a 1536-wide INT8 GEMM (the paper's
QNN analogue was 2 GiB/graph; here it is ~30× tighter). Consequences:

1. **Preset A (`M = 1530`) needs the token dimension tiled 4× (4 × 384)** on the NPU side. This is a
   *new* structural constraint the blueprint does not have: the NPU suffix must tile over
   (block, token-tile), not just over blocks.
2. **`DEAD_OBJECT` is sticky.** After the first oversized compile the HAL stops answering — subsequent
   compiles in the same session fail with `DEAD_OBJECT` even at shapes that worked before. Never
   submit a graph above the measured ceiling; probe sizes in fresh processes, ascending.
3. Effective Edge TPU throughput ≈ **285 GOPS** (0.285 TOPS) on this shape.

For scale on the same GEMM: `nnapi-reference` (CPU) takes **219 ms** at M=32 and **4.99 s** at M=1530;
the naive Mali kernel takes **338.7 ms** at M=128 (10.4 GOPS). TPU / naive-Mali = **31.6×**.

### What that predicts (arithmetic only, not yet measured)

Preset A block = `M·params_per_block = 1530 × 49,586,432` = **75.87 GMAC = 151.7 GFLOP**:

| engine rate | per block | 180 blocks (DiT only) |
|---|---|---|
| TPU @ 285 GOPS | 532 ms | **95.8 s / chunk** |
| naive Mali @ 10.4 GOPS | 14.6 s | 2,626 s / chunk |
| tiled Mali @ ~1 TOPS (hypothetical) | 152 ms | 27.3 s / chunk |

⇒ **the Mali prefix is the whole open question.** At the naive rate it can never be a stage
(`T_G ≈ 27 × T_N` ⇒ `k_b* ≈ 7`, gain ≈ 1.03×). A properly tiled INT8 GEMM would make the GPU
*comparable to or faster than* the TPU, which would invert the paper's premise and push `k_b*` late.
Kernel engineering, not hardware, decides this — measure it before any cut claim.

### Getting an NNAPI graph to validate (four traps, all hit)

1. **`FULLY_CONNECTED` takes exactly 4 inputs** — `{input, weights[N_out,K], bias[N_out], activation}`.
   Bias is mandatory; there is **no** `activationParams` operand (that belongs to other ops).
   Passing 5 operands ⇒ `model_finish` = `BAD_DATA`.
2. **You must call `addOperation`.** Operands + `identifyInputsAndOutputs` are not a model; the
   validator says `NN_RET_CHECK failed (Validation.cpp:854): !subgraph.operations.empty()`.
   **Read `adb logcat` — the NNAPI manager prints the exact failing check**; guessing wastes runs.
3. **`TENSOR_QUANT8_SYMM` is not accepted by FC.** Signed int8 is
   `ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED` with `zeroPoint = 0`. Per-output-column scales use
   `ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL` (`channelDim = 0`, operand `scale = 0`, scales via
   `setOperandSymmPerChannelQuantParams`) — the exact analogue of the paper's per-column scale vector.
4. **Weights must be a constant operand** via `setOperandValueFromMemory` from an `ANeuralNetworksMemory`
   (file-backed or AHB-backed both work). Passing weights as a plain model *input* does not validate.
   Reusing one `ANeuralNetworksExecution` for a second `compute()` returns `BAD_STATE` — create a
   fresh execution per invocation.

## Zero-copy — **Gate 2 passes on both routes**

```
IMPORT_MEMORY_SYM clGetExtensionFunctionAddressForPlatform -> found
AHB_CL  import_OK
AHB_CL  ZEROCOPY_VERIFY host[0]=23130(exp 23130) host[1]=23131(exp 23131) -> SHARED
AHB_NNAPI_MEMORY_rc=0(NO_ERROR)
MAPSHARED_OK path=/data/local/tmp/hv_shared.bin bytes=33554432
FILE_CL import_OK
FILE_CL ZEROCOPY_VERIFY host[0]=23130(exp 23130) host[1]=23131(exp 23131) -> SHARED
FILE_NNAPI_MEMORY_rc=0(NO_ERROR)
```

The GPU kernel wrote a pattern into the imported memory and the **host pointer saw it** — same
physical pages, no copy. Two working routes:

- **Route A**: `AHardwareBuffer` → `clImportMemoryARM(type=ANDROID_HARDWARE_BUFFER_ARM)` for the GPU,
  `ANeuralNetworksMemory_createFromAHardwareBuffer` for the NPU.
- **Route B (simpler, now preferred)**: one file, `mmap(MAP_SHARED)` in both processes →
  `clImportMemoryARM(type=HOST_ARM)` + `ANeuralNetworksMemory_createFromFd`. One page cache, no
  `AHardwareBuffer`, no `SCM_RIGHTS`, no fd passing. Same "one stored copy" property.

Three API traps worth remembering:

1. `dlsym(libOpenCL.so, "clImportMemoryARM")` returns NULL — Android's CL ICD loader does not forward
   vendor extension symbols. Use `clGetExtensionFunctionAddressForPlatform(platform, "clImportMemoryARM")`.
2. `cl_import_properties_arm` is **`intptr_t` (64-bit)**. Passing a 32-bit `unsigned[]` makes the driver
   read one garbage 64-bit key and answer `CL_INVALID_PROPERTY (-64)`. The list must be
   `{CL_IMPORT_TYPE_ARM, type, 0}` in `intptr_t`.
3. `AHardwareBuffer_getNativeHandle` is absent from the NDK header (see below) — but with route B you
   never need it.

## What this changes in the plan

1. **Zero-copy is solved** (route B preferred). Gate 2 can be closed as *pass* with the above evidence.
2. **Per-block AHB splitting is optional** (ceiling 1792 MiB ≫ 47.29 MiB). Keep per-block regions for
   remap granularity, but state that the constraint that forced them in the paper is absent here.
3. **The NPU graph must be token-tiled to ≤384 tokens** (4 tiles at Preset A). One graph per
   (block, token-tile), or one per block with the tile loop inside the graph if the compiler allows
   it — test both, since per-dispatch overhead is ~6 ms.
4. **Kernel quality decides whether the GPU is a stage at all.** 10.4 GOPS naive vs 285 GOPS TPU is a
   31.6× gap; a tiled GEMM must close most of it before `k_b*` means anything.

## Thermal sensing — `/sys/class/thermal` is closed, `dumpsys` is open

`cat /sys/class/thermal/thermal_zone*/temp` → **Permission denied** as the `shell` user on Android 17.
Use `dumpsys thermalservice`, which lists named zones and cooling devices:

- zones seen: `TPU`, `skin_therm1/2`, `battery`, `VIRTUAL-SKIN`, `neutral_therm`,
  `charger_skin_therm`, `quiet_therm`, `disp_therm`, `usb_pwr_therm2`, `gnss_tcxo_therm`,
  `cellular-emergency`
- cooling devices: **`tpu_cooling`**, `thermal-cpufreq-0/1/2`, **`thermal-gpufreq-0`**, `fcc`, `usbc-port`
- idle at probe time: TPU 31 °C, skin ~25 °C, battery 23 °C, Thermal Status 0
- **after the FC sweep: TPU 53 °C, skin_therm2 41.9 °C, charger_skin 45.1 °C** — i.e. the TPU zone
  is a direct, usable signal that Process B really executed on the accelerator.

⇒ `probe_host.py --cooldown` now parses `dumpsys thermalservice` (`read_zones()`), keyed on
`TPU / skin_therm1,2 / battery / VIRTUAL-SKIN`.

## Open questions this round leaves

1. **Does the physical GPU beat the TPU once the GEMM is tiled?** (decides `T_G/T_N` and thus `k_b*`)
2. **Can several ops (a whole block, or FFN up+down) live in one ≤384-token graph**, or must each
   GEMM be a separate compilation? Per-dispatch overhead ~6 ms × 40 dispatches/block would eat the
   compute budget.
3. **Does the ceiling scale with `N`?** At `N=1536, M=1530` the compile was refused like `M=448, N=8960`
   — same token limit, so it looks like an activation/token bound, but sweep `N` at fixed `M=384`.
4. **VAE decode path** — untouched so far; NNAPI has no `CONV_3D`, so this is either decomposed 2D convs
   on the TPU or OpenCL on Mali. Measure both before choosing.
