# T_N — token-tiled NNAPI block graph on the Tensor Edge TPU

Raw captures: `docs/device/npu_tn_20261115_1531{06,42,31}.json` (validation), `…_153502.json`,
`…_153652.json` (timed), `…_153817.json` (isolated `ffn.down`).
Harness: `tools/android/npublock/npu_block.cpp` · host driver `tools/npu/npu_host.py`.

**Status: T_N is still TBD.** Nine of the block's ten GEMMs are measured; the tenth
(`ffn.down`, K=8960) **cannot be compiled at all** on this HAL, and both timed runs started
above the thermal gate. What follows is what the phone actually did.

## What was submitted

One Preset-A DiT block = 10 INT8 GEMMs. Token-tiled 4× at **M = 384** (Gate 0's measured
per-graph ceiling; the binary refuses M > 384 so the sticky `DEAD_OBJECT` trap is never touched).
Weights are constant operands served from **one file-mapped INT8 blob**
(`/data/local/tmp/hv_block_int8.bin`, 46,399,488 B = the block's 10 base matrices, `createFromFd`
— route B). The blob is a deterministic LCG fill: this is a latency harness and the real INT8
artifact does not exist yet (Phase 1 outstanding), so these numbers are shape-determined. Compiles
are forced onto `google-edgetpu` alone via `Compilation_createForDevices` — no CPU fallback is
allowed to hide in the measurement.

## Measured — per GEMM, M = 384 tokens (mean of 8 samples = 4 tiles × 2 iters)

| GEMM | N_out | K | ms / token-tile | s / block (4 tiles) | GOPS |
|---|---|---|---|---|---|
| `attn1.to_q` | 1536 | 1536 | 8.098 | 0.0324 | 224 |
| `attn1.to_k` | 1536 | 1536 | 8.151 | 0.0326 | 222 |
| `attn1.to_v` | 1536 | 1536 | 8.261 | 0.0330 | 219 |
| `attn1.to_out` | 1536 | 1536 | 8.073 | 0.0323 | 225 |
| `attn2.to_q` | 1536 | 1536 | 8.868 | 0.0355 | 204 |
| `attn2.to_k` | 1536 | 1536 | 8.017 | 0.0321 | 226 |
| `attn2.to_v` | 1536 | 1536 | 8.291 | 0.0332 | 219 |
| `attn2.to_out` | 1536 | 1536 | 8.269 | 0.0331 | 219 |
| `ffn.up` | 8960 | 1536 | **36.739** | **0.1470** | **288** |
| `ffn.down` | 1536 | 8960 | **TBD (blocked: graph will not compile)** | — | — |
| **sum of the 9 measured** | | | 102.4 | **0.4112** | — |

`ffn.up` at 288 GOPS reproduces Gate 0's single-GEMM value (282 GOPS) independently, from a
different process, a different weight backing and a different code path — good corroboration that
226–288 GOPS is the real TPU rate on these shapes.

That 0.4112 s covers **70.3 %** of the block's MACs, so it is a **lower bound, not T_N**:

> `T_N > 0.4112 s/block` and `T_G = 0.1618 s/block` ⇒ **T_N / T_G > 2.54** — a measured,
> model-free inequality. The GPU is at least 2.5× faster per block than the NPU.

## Three structural findings (all new; each one costs a design decision)

### 1. A whole-block single graph **validates, is fully supported, and then reboots the phone**

The 10 chained FCs at M=384 with per-tensor scales:

```
MODEL_FINISH|nops=10|M=384|rc=0(NO_ERROR)
SUPPORT|nops=10|rc=0(NO_ERROR)          # all 10 ops: google-edgetpu reports supported
… then ANeuralNetworksCompilation_finish never returned: adb rc=255, device uptime 1 s → 29 s
```

The phone **rebooted inside `Compilation_finish`** (`init` SIGABRT in the crash buffer at the same
moment). This is the answer to Gate 0 open question #2: a whole block *can* be expressed as one
≤384-token graph — the validator and the support query both accept it — but the Coral compiler
does not survive it. **One graph per GEMM is the only path on this HAL.** The driver now records
`uptime_before/after` per mode and marks `device_rebooted`, so this can never be mistaken for a
slow run again.

### 2. Per-output-column scales are **rejected by FULLY_CONNECTED** before compilation

```
FATAL|addOperation|attn1.to_q|pc=1|in=0,1,2,3|out=4|nops_in_model=5|rc=4(BAD_DATA)
```

`TENSOR_QUANT8_SYMM_PER_CHANNEL` (channelDim=0, operand scale 0, scales via
`setOperandSymmPerChannelQuantParams`) fails **on the first operation of a one-op model**, so it is
a structural rejection, not a size or chain effect. The identical call with a per-tensor scale
succeeds. **`RESULTS_gate0.md` trap #3 asserts the per-channel form is "the exact analogue" —
that claim was never measured** (every Gate-0 FC sweep passed `per_channel = 0`) and is now
measured false.

Consequence: the locked per-output-column quantisation is still the right *storage* format (one
INT8 copy, and Mali consumes per-column scales natively), but the **NPU stage cannot consume it
with its own scales**. Options are a per-tensor-scale NPU graph (accuracy hit, must be measured in
Gate 1/4) or restricting the NPU to the shapes where an equivalent path exists. This one is a
decision for the student — it touches `design_locks.json`, so it is reported, not applied.

### 3. `ffn.down` (K = 8960) does not compile — `DEAD_OBJECT` after ~62 s

```
FATAL|comp_finish|ffn.down|ms=62047.4|rc=10(DEAD_OBJECT)     # fresh process, --only 9
FATAL|comp_finish|ffn.down|ms=65920.7|rc=10(DEAD_OBJECT)     # 10th op in a 9-op process
```

Reproduced in a fresh process, so it is the **shape**, not compile accumulation. Every Gate 0
measurement used K = 1536; the transposed FC (input width 8960, i.e. a 1536×8960 weight matrix and
384×8960 activations) never was. Decomposing K into chunks with an accumulate, or running the down
projection on Mali, is the obvious workaround — untested.

## Two measurement caveats that must survive into the paper

1. **Thermal state: neither timed run is gate-compliant.** Run `…_153502` started at TPU 67 °C,
   run `…_153652` at TPU 60 °C (battery 27.3 ⇒ gate wanted ≤ 30.3 °C). The device had just
   rebooted from finding #1 and was still hot. The two runs agree within 3 % (7.9 vs 8.1 ms per
   attn tile) and both exceed the cool-start Gate-0 rate, so they do **not** look throttled — but
   per the locked rule these numbers are **not table-grade** and need one cool re-run.
2. **`getDuration(ANEURALNETWORKS_DURATION_ON_HARDWARE)` is unusable on this HAL**: it reports
   0.006–0.031 ms, i.e. ~150 TOPS on a 0.906 GMAC GEMM. Wall time around a synchronous
   `execute` is the measurement; the hardware counter is ignored throughout.

The `TPU` thermal zone rose 31 → 60–67 °C across these runs, which is the device-level proof that
the graphs executed on the accelerator rather than the CPU.

## What is still TBD

| quantity | state |
|---|---|
| `T_N` (s/block, Preset A) | **TBD (blocked: `ffn.down` graph does not compile; 9/10 GEMMs measured, ≥ 0.4112 s/block)** |
| per-GEMM NPU rates | measured (table above), **hot-started — needs a cool re-run to be table-grade** |
| `T_V` (VAE decode) | untouched |
| co-run `T_G` | untouched |
| `k_b*` | TBD (blocked on the three above) |

## Next actions, in order

1. **Cool re-run of `--modes single`** from the idle floor (`--cooldown 30`) → gate-compliant
   per-GEMM rates. Same command, ~90 s of device time once cool.
2. **Decide the `ffn.down` path**: chunked-K FC + `ADD` (2× K=4480, or 4× K=2240) on the TPU, vs
   Mali/OpenCL for the down projection. Measure whichever is chosen.
3. **Decide the per-column-scale consequence** for the NPU stage (finding #2) — it changes Gate 1/4
   parity targets, so it must be settled before the quantiser is written.
