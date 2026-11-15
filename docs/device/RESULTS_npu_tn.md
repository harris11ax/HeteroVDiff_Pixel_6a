# T_N — token-tiled NNAPI block graph on the Tensor Edge TPU

Raw captures: `docs/device/npu_tn_20261115_153311.json` (validation + per-column rejection),
`…_153427.json` (whole-block graph, device went down), `…_153502.json` and `…_153652.json` (timed, hot),
`…_153817.json` (isolated `ffn.down`), `…_160146.json` (**timed, gate-compliant — the table numbers
below come from this one**). The `.json` files are gitignored; the `.md` siblings are
committed.
Harness: `tools/android/npublock/npu_block.cpp` · host driver `tools/npu/npu_host.py` ·
cooling gate wrapper `tools/npu/cool_rerun.py`.

**Status: T_N is still TBD.** Nine of the block's ten GEMMs are measured from a gate-compliant
cool start; the tenth (`ffn.down`, K=8960) **cannot be compiled at all** on this HAL. What follows
is what the phone actually did.

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

Capture `npu_tn_20261115_160146.json`, **cool start, gate-compliant**: before the run
TPU 29.0 °C · skin_therm1 27.6 · skin_therm2 28.4 · VIRTUAL-SKIN 27.4 · battery 26.1 ⇒ gate
`max(TPU,skin,VIRTUAL-SKIN) ≤ battery+3 = 29.1 °C`, satisfied at 29.0 °C on two consecutive samples
(start of the re-run 16:00:42, after 17 min of monotone cooling with the screen dozed and the device
network off).

| GEMM | N_out | K | ms / token-tile | s / block (4 tiles) | GOPS |
|---|---|---|---|---|---|
| `attn1.to_q` | 1536 | 1536 | 8.043 | 0.0322 | 225 |
| `attn1.to_k` | 1536 | 1536 | 8.124 | 0.0325 | 223 |
| `attn1.to_v` | 1536 | 1536 | 8.870 | 0.0355 | 204 |
| `attn1.to_out` | 1536 | 1536 | 8.659 | 0.0346 | 209 |
| `attn2.to_q` | 1536 | 1536 | 8.517 | 0.0341 | 213 |
| `attn2.to_k` | 1536 | 1536 | 8.491 | 0.0340 | 213 |
| `attn2.to_v` | 1536 | 1536 | 8.406 | 0.0336 | 216 |
| `attn2.to_out` | 1536 | 1536 | 8.692 | 0.0348 | 209 |
| `ffn.up` | 8960 | 1536 | **36.616** | **0.1465** | **289** |
| `ffn.down` | 1536 | 8960 | **TBD (blocked: graph will not compile)** | — | — |
| **sum of the 9 measured** | | | 104.4 | **0.4178** | 222 (mean) |

`ffn.up` at 289 GOPS reproduces Gate 0's single-GEMM value (282 GOPS) independently, from a
different process, a different weight backing and a different code path — good corroboration that
204–289 GOPS is the real TPU rate on these shapes.

**Cool start did not buy any speed.** The hot captures (`…_153502`, `…_153652`, TPU 60–67 °C) gave
102.4 ms/tile summed and 0.4112 s/block; this cool capture gives 104.4 ms and 0.4178 s/block — the
cool run is **1.6 % slower**, i.e. the earlier numbers were not throttled at all (within the same
3 % run-to-run spread). The gate rule still matters for the paper: this capture is the one whose
thermal state was legal, so it is the one the tables cite. Per-tile spread within this capture is
7.1–12.3 ms on the 1536-wide GEMMs (first tile of each graph carries the cold weight fetch), which
is why per-GEMM figures are means of 4 tiles and not single samples.

That 0.4178 s covers **70.3 %** of the block's MACs, so it is a **lower bound, not T_N**:

> `T_N > 0.4178 s/block` and `T_G = 0.1618 s/block` ⇒ **T_N / T_G > 2.58** — a measured,
> model-free inequality. The GPU is at least 2.6× faster per block than the NPU.

The device heated **29.0 → 59.0 °C on the TPU zone during the 54 s run** (skin 27.6 → 36.1), which is
the device-level proof that the graphs executed on the accelerator rather than the CPU.

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
384×8960 activations) never was. The cool re-run (`…_160146`) failed on the same op faster and with a
different code — `FATAL|comp_finish|ffn.down|ms=47186.8|rc=5(OP_FAILED)` — so the failure is not a
thermal artefact either; the HAL gives up on this shape somewhere between 47 s and 66 s. Decomposing K
into chunks with an accumulate, or running the down projection on Mali, is the obvious workaround —
untested.

## Two measurement caveats that must survive into the paper

1. **Thermal state: resolved — one capture is now gate-compliant.** Capture `…_160146` started at
   TPU 29.0 °C with battery 26.1 °C (gate 29.1 °C), so the table above is table-grade. Reaching the
   floor needed the device network off and the screen dozed: with the Play Store background service
   running the TPU zone sat at 44–48 °C and the gate was unreachable, and 17 min of monotone cooling
   was required (TPU 43 → 29 °C) before a legal start. The two earlier, hot captures (TPU 67 / 60 °C)
   stay in the repo as evidence that the cool run is not faster than the hot one.
   **Process note for the next device run:** `python tools/npu/cool_rerun.py` does this — it polls
   zones until `max(TPU,skin,VIRTUAL-SKIN) ≤ battery+3`, re-confirms, then passes the same threshold
   to the harness as `--cooldown`, so a timed run can never start hot.
2. **`getDuration(ANEURALNETWORKS_DURATION_ON_HARDWARE)` is unusable on this HAL**: it reports
   0.006–0.031 ms, i.e. ~150 TOPS on a 0.906 GMAC GEMM. Wall time around a synchronous
   `execute` is the measurement; the hardware counter is ignored throughout.

The `TPU` thermal zone rose 31 → 60–67 °C across these runs, which is the device-level proof that
the graphs executed on the accelerator rather than the CPU.

## What is still TBD

| quantity | state |
|---|---|
| `T_N` (s/block, Preset A) | **TBD (blocked: `ffn.down` graph does not compile; 9/10 GEMMs measured, ≥ 0.4178 s/block)** |
| per-GEMM NPU rates | **measured, gate-compliant** (`…_160146`, cool start at TPU 29.0 °C) |
| `T_V` (VAE decode) | untouched |
| co-run `T_G` | untouched |
| `k_b*` | TBD (blocked on the three above) |

## Next actions, in order

1. ~~Cool re-run of `--modes single` from the idle floor~~ — **done 2026-11-15 16:00:42**, capture
   `…_160146`, gate-compliant; the table above is from it. Tool: `tools/npu/cool_rerun.py`.
2. **Decide the `ffn.down` path**: chunked-K FC + `ADD` (2× K=4480, or 4× K=2240) on the TPU, vs
   Mali/OpenCL for the down projection. Measure whichever is chosen. This is the only thing between
   the current 0.4178 s lower bound and a real `T_N`.
3. **Decide the per-column-scale consequence** for the NPU stage (finding #2) — it changes Gate 1/4
   parity targets, so it must be settled before the quantiser is written.
