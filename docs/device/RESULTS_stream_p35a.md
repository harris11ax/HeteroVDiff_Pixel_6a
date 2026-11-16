# P3.5a — the cut's real cadence: one handoff per 160 GPU blocks

*2026-11-16 00:0x. Tool: `phase3_pipeline/stream.py`. Capture
`stream_20261115_233709` (JSON + raw log + this run's `stream_*.md`).
Start was **table-grade**: TPU 25.0 °C, VIRTUAL-SKIN 23.3 °C, battery 22.3 °C against a
25.3 °C gate (peak ≤ battery + 3).*

P3.4 closed the coordinator but found that the GPU stage published **per block** while the
cut hands off **once per 160**, so the GPU finished its work in 0.214 s and idled 18.4 s
(overlap 1.1 %). It also found a ~80 ms/block dead gap from the NPU harness hashing `ffn.up`'s
13.7 MB output every block. This run fixes both and measures the stage at the shape the cut
actually uses:

    hv_proc_a --blocks 320 --pub-every 160 --once-blocks 4   # 2 chunks x 160 GPU blocks
    hv_proc_b --blocks 2 --nblocks 20 --hash-every 20        # 2 chunks x 20 NPU blocks

Both changes are arguments, not behaviour changes: `--pub-every 1` and `--hash-every 1`
reproduce P3.4 exactly. Slot and sequence numbering now follow the **publish ordinal**, so the
ring protocol (`done >= seq - SLOTS`) is untouched.

## 1. Verdict

**PASS.** GPU published **2/2** chunks over **320 blocks**, NPU consumed **2/2**,
**1440/1440** held-graph executes, 10/10 weight windows imported zero-copy, one
`createFromFd` view on the NPU side (one stored INT8 copy), NaN words **0**, fatal **0**,
wall **80.1 s**.

| quantity | value |
|---|---|
| GPU block, 10 per-GEMM closures (route `import`) | **202.1 ms** median, 173.3–232.1 (n=**320**) |
| GPU block, one closure (`TSC|gpu`, reference form) | **166.2 ms** median, 165.3–171.4 (n=4), 854.2 GOPS |
| GPU throughput, 10-closure form | 702.6 GOPS |
| NPU block on the TPU (9 GEMMs, 4 tiles) | **392.8 ms** median, 384.6–432.3 (n=40) |
| NPU chunk window (quant + 20 blocks) | **7.93 s / 7.86 s** |
| GPU publish interval (per 160-block chunk) | **32.544 s** |
| publish→consume | **0.20 / 0.45 µs** (mean 0.33 µs) |
| GPU slot wait (back-pressure) | **0 µs** — the ring never binds at this shape |
| payload conversion (host-side fp16) | 48.7 ms per chunk (max 51.5) |
| `msync` (first page) | max 4.54 ms |
| `ffn.up` output hash cost | **170.9 ms over 2 calls — 85.5 ms each** |
| both stages executing at once | **7.93 s = 50.2 % of NPU busy** |

## 2. The two P3.4 defects are closed, with numbers

- **Publish cadence** — the GPU now runs 160 blocks between handoffs and publishes 2 chunks in
  one continuous stream. Publish interval 32.544 s, back-pressure **0 µs** (P3.4: producer
  blocked 220 ms with a consumer 2× slower per chunk). Overlap of NPU busy rose **39.4 % →
  50.2 %**, and it is now capped by a different thing: the GPU is the long pole by **4.1×**
  (32.5 s of GPU work per 7.9 s of NPU work), so the NPU idles ~76 % of the period. That is a
  property of the cut at `k_b = 160`, not of the handoff.
- **Hash gap** — hashing every 20th NPU block instead of every block costs **170.9 ms total**
  in this run instead of **3.4 s** (40 blocks × 85.5 ms). The measured per-call cost, 85.5 ms,
  confirms the ~80 ms/block dead gap P3.4 inferred. `--hash-every 1` restores the old
  behaviour; Gate 3 parity does not depend on hashing every block.

**Read the 50.2 % correctly.** It is *one* of the two chunk windows: the last chunk's handoff
necessarily happens at the end of the GPU stream, so chunk 2's NPU work (7.86 s) runs after the
GPU has finished and cannot overlap anything. `both_busy` = 7.93 s is chunk 1's window, fully
inside the GPU stream. In steady state each chunk's NPU work overlaps the *next* chunk's GPU
work, so 2 chunks structurally understates the overlap — the steady-state number needs P3.5's
10 chunks, not a longer argument.

Both payloads hash to the same FNV value (`15529b56755822dc`, 6 zero words, 0 NaN, per chunk)
because `proc_a` runs the same fixed activation buffers for every block — the harness has no
time-varying input per block. That is why the two chunks are byte-identical; it is not a stale
slot, and `proc_b` released each slot before the next publish.

## 3. The period is 32.54 s, not the predicted 23.61 s — and the gap decomposes

The pipeline's measured chunk period at the locked cut (`k_b = 160`) is **32.544 s**:
**1.38×** the 23.61 s that `configs/rates_preset_a.json` predicts. The decomposition is
measured, not modelled:

| form | GPU ms/block | 160-block stage | vs 23.61 s |
|---|---|---|---|
| co-run solos, staging, `k_b*` input (`T_G` = 0.1474) | 147.4 | 23.59 s | 1.00× |
| one closure per block, **import** route (this run) | 166.2 | 26.59 s | **1.13×** |
| 10 closures per block + TS stamps, import (this run) | 202.1 | 32.34 s | **1.37×** |

So of the 9.0 s excess: **+3.0 s is the zero-copy import penalty** (P3.2 measured 214.5 vs
162.2 ms/block; here 166.2 vs the 147.4 staging co-run) and **+5.7 s is the harness's
per-GEMM completion barriers** — the 10-closure form exists to stamp each GEMM, which is how
the per-GEMM table was produced, and it is not how the stage has to run.

**Cut consequence, reported not applied** (`solve_cut.py`, `T_N` = 0.6484, `T_V` = 10.640,
`B` = 180):

| `T_G` | `k_b*` | period | speedup vs NPU-only |
|---|---|---|---|
| 0.1474 (locked) | **160** | 23.61 s | 5.394× |
| 0.1662 (import, one closure) | **157** | 26.09 s | 4.881× |
| 0.2021 (import, 10 closures) | **150** | 30.32 s | 4.201× |

The locked `k_b* = 160` is only correct on the staging+co-run pair. With the pipeline as built
it costs a 32.54 s period. Two independent levers, both already measured, move it back:
the closure overhead (-5.7 s) is removable by closing timings once per block, and the import
penalty (+3.0 s) is P3.2's staging-vs-import trade. Applying either changes
`configs/rates_preset_a.json`, which is locked — this is a decision for the student.

## 4. Drift and thermal over 64.7 s of continuous GPU work

320 consecutive blocks, first-20 % **206.7 ms** vs last-20 % **206.0 ms** = **−0.34 %**;
slope flat, no derating. Zones: TPU 25.0 → **45.0 °C**, VIRTUAL-SKIN 23.3 → 30.8 °C,
skin 23.5 → 34.6 °C, battery 22.3 → 22.9 °C. Start was inside the gate, so the per-block rate
is **table-grade** even though the phone ends 20 °C hotter on the TPU zone: the GPU stream
itself shows no throttle term, matching the 24.6 s NPU soak's conclusion for the other engine.

## 5. What this proves / does not

- proves: the two-process pipeline runs the cut's real 160/20 shape end-to-end on the phone
  with **one handoff per chunk**, sub-microsecond handoff latency, no back-pressure, and a
  steady 202.1 ms GPU block over 320 blocks with no drift
- proves: the two defects P3.4 named are closed by argument, and both leave P3.4's behaviour
  available (`--pub-every 1`, `--hash-every 1`)
- does **not** prove: the 10-chunk steady state (P3.5 — this is 2 chunks), numerical parity
  (Gate 3), or the decode (still a cited 10.640 s stub)
- does **not** prove: byte parity across the handoff. `proc_b` prints its own FNV chain
  (`39c12d9adfc0cd13` over 2 payloads); with `--source none` the producer prints no comparable
  chain, so integrity here is the ring protocol plus the P3.1 primitive result, not a chain match
