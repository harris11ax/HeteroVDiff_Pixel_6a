# HANDOFF — HeteroVDiff on Pixel 6a

*Updated 2026-11-16 (00:40, **P3.6 landed — Gate 4 PASS, one-copy proven, memory 0.27 GiB**). This
file is the orders sheet: read it first, then `docs/00_START_HERE.md`. The pipeline detail is
`docs/device/RESULTS_stream_p35.md` (run `stream_20261116_000415`); Gate 4 is
`docs/device/RESULTS_gate4_p36.md` (runs `gate4_20261116_003217` / `…_003411`).*

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
| **VAE decode (`T_V`)** | **both branches measured** → `docs/device/RESULTS_vae_tv.md`; TPU (36 causal convs as `kT` shifted `CONV_2D` taps) **10.554 s/chunk**, 969.7 GOPS; Mali (`conv8_taps`, implicit GEMM, one dispatch per class) **12.256 s/chunk**, 417.5 GOPS, 7/13 classes exact vs the int32 reference. **TPU 1.16× faster** on the convs — the decode ordering is the opposite of the block's. Mid-block attention (32.4 GMAC) **now measured on both engines**: TPU **0.601 s**, Mali **0.0855 s** (one dispatch per GEMM) ⇒ **Mali 7.0× faster**, the opposite ordering again. `T_V` = **11.155 s all-TPU · 10.640 s hybrid · 12.342 s all-Mali** — no longer a lower bound |
| Gate 1 (artifact) | **INT8 base + verifier done** → `docs/device/RESULTS_gate1.md`; 1,391,984,640 B INT8 (exact), 300/300 GEMMs cosine ≥ 0.999922, coverage 1317/1317 tensors. **Outstanding: end-to-end PSNR/SSIM leg — no longer environment-blocked: `.venv-ref` (torch 2.15.0+cpu, diffusers 0.43.0) is installed, commit `8663c3f`** |
| Gates 3–5 | pending |
| **Co-run `T_G` / `T_N`** | **measured** → `docs/device/RESULTS_corun_tg.md`; both engines executing at once, 99.8 % of GPU-time overlapped NPU executes. GPU base block **0.1474 s** (no slowdown — it measured 8 % *faster* than solo), NPU 9-GEMM round **0.3904 s** (±4 % per GEMM). `T_N/T_G = 4.40×` |
| **Sustained `T_N`** | **measured** → `docs/device/RESULTS_sustain_npu.md`; 60 rounds × 9 GEMMs = **24.6 s of continuous TPU work, drift +0.35 %** (first-20 % 408.9 / last-20 % 410.3 ms per round, slope +0.03 ms/round) while the TPU zone rose 25.0 → 31.0 °C ⇒ **no derating over a full NPU stage**, so `T_N` carries no throttle term. Tool: `python tools/npu/sustain_npu.py` |
| **`k_b*`** | **solved = 160** → `configs/rates_preset_a.json` + `phase2_profile/solve_cut.py`; period 23.61 s, 5.394× vs NPU-only, GPU `[0,160)` + NPU `[160,180)` + decode. Survives the attention term **only if the attention runs on Mali** (hybrid decode, `T_V` = 10.640 s); all-TPU decode moves it to 161 |
| **Gate 4 (memory + one copy)** | **PASS twice** → `docs/device/RESULTS_gate4_p36.md`, tool `tools/pipe/gate4.py`, captures `gate4_20261116_003217` / `…_003411`. PSS A+B peak **282.8 MB (0.27 GiB) vs the 2.2 GiB budget — 8.2× headroom** (A 246.9 MB median, B 30.9 MB). `proc_a`: **one `rw-s` MAP_SHARED view** of the 46,399,488 B block at offset 0, **10/10** windows imported by reference. `proc_b`: one fd, handed to `createFromFd`, **0 mappings** of its own. **Trap: `smaps` under-reports the GPU stage 2.93×** (`Pss` 85,137 kB vs `dumpsys` TOTAL 249,290 kB — the 164,192 kB `GL mtrack` row is driver-accounted CL memory and is absent from `/proc/<pid>/smaps`) ⇒ quote `dumpsys`, never `smaps`, for Mali |
| Blocker | every rate the cut needs is now measured on both engines **and checked for sustained drift** — the mid-block VAE attention closed 2026-11-15 21:2x, the 24.6 s sustained soak 21:41. What is still unrun: the Gate-1 end-to-end PSNR/SSIM leg and `calibrate.py` — **environment resolved 2026-11-15 (commit `8663c3f`): use `.venv-ref/Scripts/python.exe`** (torch 2.15.0+cpu, diffusers 0.43.0, transformers 5.23.0, deliberately NOT `.venv-quant` because this cron fires every 15 min in this tree). Measured fp32 GEMM 4096³ = 449.9 GFLOPS, so a 36.0 TFLOP 9-frame reference chunk is ~1.3 min. **Gate 4 is built and PASSing**; what remains of Gates 3–5 is Gate 3's numerical parity, the decode as an execution, and Gate 5 |

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
   **NEXT ACTION (P3.7): the decode as an execution + Gate 3's parity.** `--decode 1` is
   still a no-op stub and `proc_b` only *hashes* the payload it consumes. Wire the measured
   10.554 s TPU decode into the NPU stage as real work (that also makes the period at
   `k_b` = 160 a full-cut number instead of GPU-plus-20-blocks), and compare one block's
   INT8 output against an int32 reference computed on the host, so Gate 3 has a number
   rather than a hash. Gate 1's end-to-end PSNR/SSIM leg and `calibrate.py` remain
   host-only work for a tick when the phone is absent or hot.
   Host-only work that can run in parallel: ~~fold `T_V` = 10.640 (hybrid) into `configs/rates_preset_a.json`~~
   — **done 2026-11-15 21:3x** (k_b* re-solved: 160, period 23.61 s).
10. Phase 4 tables, report.

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

## Definition of done for the course project

1. Artifact + quantiser + parity report (Gate 1) committed.
2. Device probe + measured rates + solved cut (Gate 2) committed. ← partially: GEMM done
3. Two-process pipeline streaming ≥10 chunks on the phone without manual intervention (Gates 3–5).
4. `results/tables.md`: configurations, cut ablation, memory, quality (PSNR/SSIM/LPIPS), SLO sweep,
   plus a prediction-vs-measured table from Phase 2.
5. Report-ready list of deviations from the paper, each with the measurement that forced it.
