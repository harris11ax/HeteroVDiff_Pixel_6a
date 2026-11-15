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

## Zero-copy routes

| route | result |
|---|---|
| `AHardwareBuffer` → `ANeuralNetworksMemory_createFromAHardwareBuffer` | **rc=0 (NO_ERROR)** |
| file (`MAP_SHARED`) → `ANeuralNetworksMemory_createFromFd` | **rc=0 (NO_ERROR)** |
| `AHardwareBuffer` → `clImportMemoryARM` | resolves; first attempt `err=-64` = `CL_INVALID_PROPERTY` because the properties list was not zero-terminated — **fixed, re-test pending** |
| file mmap → `clImportMemoryARM` (`CL_IMPORT_TYPE_HOST_ARM`) | same, pending |

**How to get `clImportMemoryARM`**: `dlsym(handle_of_libOpenCL.so, "clImportMemoryARM")` returns
NULL — Android's CL ICD loader does not forward vendor extension symbols, and
`/vendor/lib64/libOpenCL-pixel.so` does not export it either. The working path is
`clGetExtensionFunctionAddressForPlatform(platform, "clImportMemoryARM")` (CL 1.2+): **found**.
Record this in the report; it is the kind of thing that costs a week if discovered late.

## NNAPI — the Edge TPU *is* reachable

```
NNAPI_DEVICE_COUNT_rc=0 count=2
NNAPI_DEV[0].NAME=google-edgetpu     TYPE=4 (ACCELERATOR) fl=1.3
NNAPI_DEV[1].NAME=nnapi-reference    TYPE=2 (CPU)          fl=1.3
```
`google-edgetpu` **reports support for `FULLY_CONNECTED`** (`getSupportedOperationsForDevices` → true)
and **executes** an INT8 FC (`32 × 1536 × 8960`) — wall **5.1–5.9 ms**, i.e. ~150 GOPS effective on
that shape, vs 10.4 GOPS for the naive Mali kernel and ~4 ms…s for the CPU reference path.
For scale: the same FC on `nnapi-reference` takes **219 ms** (M=32) and **4.99 s** (M=1530).

### Getting an NNAPI graph to validate (three traps, all hit)

1. **`FULLY_CONNECTED` takes exactly 4 inputs** — `{input, weights[N_out,K], bias[N_out], activation}`.
   Bias is mandatory; there is **no** `activationParams` operand (that belongs to other ops).
   Passing 5 operands ⇒ `model_finish` = `BAD_DATA`.
2. **You must call `addOperation`.** Building operands and `identifyInputsAndOutputs` is not a model;
   the validator says `NN_RET_CHECK failed (Validation.cpp:854): !subgraph.operations.empty()`.
   **Read `adb logcat` — the NNAPI manager prints the exact failing check**; guessing wastes runs.
3. **`TENSOR_QUANT8_SYMM` is not accepted by FC.** Signed int8 is
   `ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED` with `zeroPoint = 0`. Per-output-column scales use
   `ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL` (`channelDim = 0`, operand `scale = 0`, scales
   passed via `setOperandSymmPerChannelQuantParams`) — that is the exact analogue of the paper's
   per-column INT8 scale vector.

Weights should be a **constant operand** supplied by `ANeuralNetworksModel_setOperandValueFromMemory`
from an `ANeuralNetworksMemory` (this is also how the shared artifact reaches the NPU): the same
model with weights as a plain *input* did not validate.

## Thermal sensing — `/sys/class/thermal` is closed, `dumpsys` is open

`cat /sys/class/thermal/thermal_zone*/temp` → **Permission denied** as the `shell` user on Android 17.
Use `dumpsys thermalservice`, which lists named zones and cooling devices:

- zones seen: `TPU`, `skin_therm1/2`, `battery`, `VIRTUAL-SKIN`, `neutral_therm`,
  `charger_skin_therm`, `quiet_therm`, `disp_therm`, `usb_pwr_therm2`, `gnss_tcxo_therm`,
  `cellular-emergency`
- cooling devices: **`tpu_cooling`**, `thermal-cpufreq-0/1/2`, **`thermal-gpufreq-0`**, `fcc`, `usbc-port`
- at probe time: TPU zone 31 °C (45 °C cached), skin ~25 °C, battery 23 °C, Thermal Status 0

⇒ the thermal gate must parse `dumpsys thermalservice` (the `TPU` zone is the one that matters for
Process B, `thermal-gpufreq-0`/skin for Process A), and `probe_host.py --cooldown` needs updating to
read it instead of `/sys/class/thermal`.

## What this changes in the plan

1. **Per-block AHB splitting is optional here** (ceiling 1792 MiB ≫ 47.29 MiB). Keep per-block regions
   for remap granularity, but state that the constraint that forced them in the paper is absent.
2. **Zero-copy is plausible on both sides** (AHB→NNAPI works; AHB→CL resolves). Re-test after the
   properties fix before committing to route A; route B (file mmap) also validated on the NNAPI side.
3. **`T_N` is obtainable** — the Edge TPU compiles and runs DiT-shaped INT8 FCs. Good news; the
   open question is the *maximum* shape it will compile (M sweep running) and how it behaves when
   several FCs are chained into a block.
4. **Kernel quality dominates `T_G`.** 10.4 GOPS naive vs ~150 GOPS TPU on the same GEMM means a
   naive Mali prefix would look useless. Write the tiled GEMM before believing any `T_G`.
