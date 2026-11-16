# T_N — token-tiled NNAPI block graph on the Tensor Edge TPU

Raw captures: `docs/device/npu_tn_20261115_153311.json` (validation + per-column rejection),
`…_153427.json` (whole-block graph, device went down), `…_153502.json` and `…_153652.json` (timed, hot),
`…_153817.json` (isolated `ffn.down`), `…_160146.json` (**timed, gate-compliant — the table numbers
below come from this one**). The `.json` files are gitignored; the `.md` siblings are
committed.
Harness: `tools/android/npublock/npu_block.cpp` · host driver `tools/npu/npu_host.py` ·
cooling gate wrapper `tools/npu/cool_rerun.py`.

**Status: `T_N = 0.6758 s/block`, measured as the sum of two measured parts** (§4). Nine of the
block's ten GEMMs are measured from a gate-compliant cool start; the tenth (`ffn.down`, K=8960) has
**no compilable graph as stored** — the compiler's ceiling is K ≤ 1664 — and is measured instead as
7 × FC`[1536,1280]` + 6 × `ADD` (0.2580 s/block, 5.45× the Mali time for the same shape). What follows
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

## 4. `ffn.down` — the compiler's **K ceiling**, and the chunked path (new)

Raw captures: `npu_tn_20261115_172850.json` (validation probes), `…_173007.json`
(`down2` in one graph), `…_173820.json` (K sweep) — the sweeps below are from the
`ksweep` / `kcompile` / `down2` / `downsep` modes added to `npu_block`.

**The wall is the reduction dimension, not the weight size.** One `FULLY_CONNECTED` of
shape `[N_out=1536, K]`, per-tensor scales, `M=384`, forced onto `google-edgetpu`, one
fresh process per K:

| K | `model_finish` | compile |
|---|---|---|
| 1280 | NO_ERROR | **OK, 0.34 s** |
| 1536 | NO_ERROR | **OK, 0.51 s** |
| 1600 | NO_ERROR | **OK, 0.46 s** |
| 1664 | NO_ERROR | **OK, 0.52 s** |
| 1792 | NO_ERROR | OP_FAILED after 48.7 s |
| 2240 | NO_ERROR | OP_FAILED after 46.0 s |
| 2560 | NO_ERROR | OP_FAILED after 59.4 s |
| 4480 | NO_ERROR | OP_FAILED after 47.8 s |
| 8960 (`ffn.down` as stored) | NO_ERROR | OP_FAILED 47.2 s / DEAD_OBJECT 62.0 s |

`ffn.up` — the *same* 13.7 MB of weights, N=8960, K=1536 — compiles in **1.44 s**. So the
compiler gives up between K = 1664 and K = 1792 regardless of N or of how many weight bytes
the operand holds, and it burns 46–59 s doing it. `getSupportedOperationsForDevices` reports
**every** shape above supported, including K=8960, so the support query is not a viability
test — the same lesson as the whole-block reboot, now measured a second time.

**Chunking inside one graph does not clear the ceiling.** `down2` = 2 × FC`[1536,4480]` +
`ADD`, one model: validates (`rc=0`), all 3 ops report supported, and then
`COMPILE|mode=down2|Kc=4480|ms=47750.0|rc=5(OP_FAILED)`. The limit is per-operation, so a
chunk must be **≤ 1664 wide and its own compilation**.

**The split that does compile — `downsep`: 7 × FC`[1536,1280]` + 6 × `ADD`, one compilation
per op.** Compile time total **2.57 s** (7 FCs at 0.34–0.38 s, one ADD at 0.09 s). Per token
tile the sequence is 13 dispatches (7 FC + 6 ADD); per block 4 tiles = 52 dispatches, the same
order of dispatch overhead the `single` fallback already accepted (40/block).

| `ffn.down` per Preset-A block | measured | source |
|---|---|---|
| **Mali-G78** (tiled `64×64×32`, tile 8×4) | **47.33 ms** (890 GOPS) | `RESULTS_gemm_mali.md` §3–4, `gemm_mali_20261115_152103.json` |
| **Edge TPU**, single FC K=8960 | **no graph** — never compiles | this section |
| **Edge TPU**, `downsep` 7×K=1280 | **64.50 ms / token-tile → 0.2580 s/block** (163.9 GOPS) | `npu_tn_20261115_174838.json` |

The `downsep` capture started at **TPU 29.0 °C, battery 25.4 °C** — the TPU zone had been flat at
29.0 °C for four consecutive samples (the idle floor; the battery itself had cooled to 25.4, so the
`battery+3` gate sat at 28.4 and the run is **0.6 °C above it** rather than under it). That is the
same floor the table-grade capture sat on (29.0 vs 29.1); it is not a hot start. The 3.2 s run
moved the TPU zone 29.0 → 32.0 °C. Per-tile samples: 70.23 / 62.74 / 62.51 / 65.05 ms (first tile
carries the cold weight fetch), mean of 8.

**Branch comparison for the student's decision** (`ffn.down`, one Preset-A block):

| path | per block | vs Mali | state |
|---|---|---|---|
| Mali-G78 tiled INT8 | **47.33 ms** | 1.00× | measured, `RESULTS_gemm_mali.md` |
| Edge TPU, single FC | — | — | **structurally impossible**: K=8960 > the 1664 compiler ceiling |
| Edge TPU, 7×K=1280 + 6 ADD | **258.0 ms** | **5.45× slower** | measured, `…_174838` |

So the TPU *can* host the down projection only by paying 5.45× the GPU's time for it, and the
margin is not a dispatch-overhead artefact: 13 dispatches per token tile at ~0.5 s of total compile
against 48 ms of compute. The engineering conclusion is that `ffn.down` belongs on Mali whichever
way the cut lands; the decision is the student's, the numbers are these.

**`T_N` is now measured — as the sum of the two measured parts** (per-GEMM table from `…_160146`,
chunked down from `…_174838`):

> `T_N = 0.4178 + 0.2580 = 0.6758 s/block` (70.99 GMAC ⇒ **210 GOPS**) ⇒
> `T_N / T_G = 0.6758 / 0.1618 = ` **4.18×** — the GPU is 4.2× faster per block, measured.
> Caveat for the paper: this is a measured **sum**, not one co-run of the whole block. The stage
> rate the cut is allowed to use still has to come from the co-run measurement (action 5).

Two more validator facts worth keeping, both found while building the probes:

- An `ADD` input operand that no operation produces and no value backs fails `model_finish`
  with `BAD_DATA` (`operandValueKnown[i] Operand N is never written`). Allocate the fused
  activation operand once and pass **existing** operand ids for both inputs.
- Fan-out of one model input into two FCs validates fine. The 10-op chain's reboot is a
  *compiler* limit, not a graph-shape one.

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
| `T_N` (s/block, Preset A) | **0.6758 s/block = 0.4178 (9 GEMMs, per-GEMM table) + 0.2580 (`ffn.down`, 7×K=1280 chunked)**; a measured **sum**. Superseded for the cut by the co-run **0.6484 s** (`RESULTS_corun_tg.md`) |
| per-GEMM NPU rates | **measured, gate-compliant** (`…_160146`, cool start at TPU 29.0 °C) |
| `ffn.down` chunked path | **measured** (`…_174838`, floor start at TPU 29.0 °C, 0.6 °C over a battery-driven gate) |
| `T_V` (VAE decode) | **measured on both branches** → `RESULTS_vae_tv.md`, `RESULTS_corun_tg.md` |
| co-run `T_G` | **measured 2026-11-15 19:48** → `RESULTS_corun_tg.md`: GPU base block **0.1474 s** while the TPU executed (no slowdown; 8 % faster than solo), NPU 9-GEMM round **0.3904 s** (per-GEMM Δ ≤ 4 %); `T_N` co-run = **0.6484 s** (downsep part solo) |
| `k_b*` | **solved = 160** (`configs/rates_preset_a.json`, period 23.58 s, 5.40× vs NPU-only); caveat: `T_V` is still a lower bound (mid-block attention unmeasured), and 162 with the Mali decode |
| per-tensor vs per-column scales (parity) | **measured** → `RESULTS_scales.md`: 40 GEMM instances, blocks 0/5/15/29, real FP32 weights. Per-tensor mean GEMM output error **3.01 %** vs per-column **1.40 %** (2.15×; worst 5.44×), Gate-1 weight cosine ≥ 0.999 fails 5/40 vs 0/40; penalty tracks column-scale spread r = 0.805. No device latency involved |

## Next actions, in order

1. ~~Cool re-run of `--modes single` from the idle floor~~ — **done 2026-11-15 16:00:42**, capture
   `…_160146`, gate-compliant; the table above is from it. Tool: `tools/npu/cool_rerun.py`.
2. ~~Decide the `ffn.down` path~~ — **both branches measured 2026-11-15 17:48** (§4): Mali
   47.33 ms/block vs Edge TPU 258.0 ms/block as 7×K=1280 + 6 ADD; a single FC at K=8960, 4480 or
   1792 has no compilable graph (K ceiling 1664). Mali wins by 5.45×, so the TPU cannot host the
   down projection — **the choice is the student's, the comparison is in §4**.
3. ~~**`T_V` — VAE decode.**~~ **done 2026-11-15** → `RESULTS_vae_tv.md`: shifted-`CONV_2D` taps on
   the TPU **10.554 s/chunk** vs Mali `conv8_taps` **12.256 s/chunk** ⇒ the NPU is 1.16× faster at
   the decode. Open ends: mid-block 3-D attention (32.4 GMAC) has no rate on either engine, so
   `T_V` is a lower bound; and the Mali conv runs at 48 % of its own GEMM rate (tile re-staged per
   tap) — a halo-staged tile is the next measurement.
4. **Co-run `T_G`.** ~~Two processes, one file-mapped artifact, GPU on blocks `[0,k)` while the NPU
   runs blocks `[k,180)` + decode.~~ **done 2026-11-15 19:48** → `RESULTS_corun_tg.md`: 99.8 % overlap,
   GPU base block 0.1474 s (solo 0.1618 — no contention measured), NPU 9 GEMMs 0.3904 s (solo table
   0.4178), so this section's `T_N` of 0.6758 s becomes **0.6484 s** under the co-run and `k_b* = 160`.
   Two open ends: `ffn.down`'s 0.2580 s is still a solo number, and the GPU got *faster* under load
   for a reason this run did not establish (DVFS is the candidate, not a claim).
5. **Decide the per-column-scale consequence** for the NPU stage (finding #2) —
   **cost measured 2026-11-15** → `docs/device/RESULTS_scales.md` + `phase1_quant/scale_parity.py`
   (40 GEMM instances, blocks 0/5/15/29, real FP32 weights, float64 reference): per-tensor scales
   cost **2.15× the mean GEMM output error** (3.01 % vs 1.40 %; worst instance 5.44×) and fail the
   Gate-1 `cosine ≥ 0.999` weight bar on **5 of 40** instances (worst 0.996892) where per-column
   fails none. The penalty tracks column-scale spread (**r = 0.805**), so it is a weight property,
   not a shape property — a shape-only "equivalent path" restriction is not supportable, and
   per-tensor-where-cheap would cover only `attn2.to_k`+`attn2.to_v` = **10.2 % of block MACs**.
   **The choice stays the student's; both branches now have numbers.**

6. **`T_N` on the exact recipe — done in two steps.** **P3.7c (2026-11-16 01:36): solo cost.**
   The only graph geometry this HAL returns bit-exactly (P3.7b: `N ∈ {128,384,768}` × `M ≤ 200`)
   costs **0.7164 s/block** for the nine GEMMs — **1.90× the 377.8 ms 9-graph stage, at 140 GOPS
   instead of 265**, **56 compilations (8.64 s) / 448 executes** per 1536-token block — and
   returns **448/448 tiles bit-exact (Δ = 0 LSB)** including `ffn.up`'s tail-128 block, verified
   against the raw weight bytes. Four runs, 0.3 % spread. → `docs/device/RESULTS_tn_recipe.md`,
   tool `tools/pipe/tn_recipe.py`, capture `tn_recipe_20261116_013625`.
   **P3.7d (2026-11-16 02:09): the co-run treatment, because 0.7164 was solo.** → 
   `docs/device/RESULTS_corun_recipe.md`, tool `tools/corun/co_recipe.py`, capture
   `corun_recipe_20261116_020926`. Under GPU load the recipe costs **700.5 ms/block** vs
   **707.0 ms** before the GPU joined in the same run (**−0.9 %: contention is not measurable**),
   while the GPU block reads **146.05 ms** contended vs **160.27 ms** solo (the same "faster under
   load" direction as action 4; cause still DVFS-unestablished). `T_N` = 0.7005 + 0.2580 downsep
   (**still solo**) = **0.9585 s** ⇒ `k_b* = 166`, period **24.25 s (+2.7 %)** — **reported, not
   applied**. To make a pass classifiable, `run_recipe` is now **pass-major** (all 56 graphs
   compiled up front, then each pass walks the block); sanity run `tn_recipe_…_020353` reproduces
   710.3 ms solo. **`ffn.down` under contention — done 2026-11-16 03:47** (`RESULTS_corun_recipe.md` §P3.7j):
 62.769 ms/tile solo ⇒ 0.2511 s/block vs 63.334 contended ⇒ 0.2533 s (+0.90 %, inside the arm
 spread) — no contention penalty. `T_N` exact is now co-run end to end: **0.8905 s/block** ⇒
 `k_b*` 165, 24.21 s, reported not applied. **448/448 exactness under the pass-major binary is
 also established** (P3.7e, `tn_recipe_20261116_023152`, Δ = 0 LSB). **This line is closed**; the
 live next actions are in `HANDOFF.md` (SA → device, `--decode 1`, Gate 5).
