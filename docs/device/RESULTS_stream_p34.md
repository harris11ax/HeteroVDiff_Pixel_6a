# P3.4 — coordinator: two stages, one ring, measured overlap

*2026-11-15 23:1x. Tool: `phase3_pipeline/stream.py`. Captures:
`stream_20261115_231417.json` (4 chunks × 1 NPU block) and
`stream_20261115_231451.json` (2 chunks × 20 NPU blocks — the cut's real stage).*

The P3.1 primitive, the P3.2 GPU stage and the P3.3 NPU stage are now driven by one
coordinator in **one `adb shell`**: it gates on thermals, pushes what changed, launches
`proc_b` (consumer) and then `proc_a` (producer) against one ring file, collects both
stages' `TS|` streams, and computes the per-chunk period, publish→consume, both engines'
busy time and the fraction of time both engines were executing at once.

Both runs **PASS** on the protocol: 4/4 and 2/2 chunks published, 4/4 and 2/2 consumed,
144/144 and 1440/1440 held-graph executes, 10/10 weight windows imported zero-copy on the
GPU side, one `createFromFd` view on the NPU side, zero NaN words, no fatal.

## 1. A launch-order rule that is a correctness requirement, not a style choice

Both binaries call `hv::reset_headers()` when they map the ring. Launched together, the
consumer can wipe a publish the producer has already made — which shows up 300 s later as
`FATAL|chunk_never_published`, i.e. as a mysterious hang rather than as a bug. The runner
therefore starts the consumer, waits for its `READY` line (graphs compiled, ring mapped,
headers reset: 33 polls ≈ 3.3 s), and only then starts the producer. Measured that way, the
first chunk's publish→consume is **0.244 µs** — the handoff primitive itself, uncontended.

## 2. Four chunks × 1 NPU block (bounded proof)

| quantity | value |
|---|---|
| wall (whole `adb shell`) | **8.2 s** |
| GPU block, 10 per-GEMM closures (route `import`) | **187.2 ms** median, 183.3–215.3 (n=4) |
| GPU block, one closure (`TSC|gpu`, reference form) | **168.0 ms** median, 164.1–169.6 |
| GPU payload convert (host-side fp16) | 25.1 ms median (12.3–44.6) |
| NPU block on the TPU (9 GEMMs, 4 tiles) | **385.9 ms** median, 374.7–393.1 |
| NPU chunk window (quant + block) | 397.0 / 389.5 / 402.8 / 411.1 ms |
| payload quantisation (fp16→INT8) | 6.4 ms median |
| GPU publish offsets from chunk 1 | 0 / 0.234 / 0.634 / 1.055 s |
| publish→consume, chunk 1 (primitive, uncontended) | **0.244 µs** |
| publish→consume, chunks 2–4 (consumer mid-block) | median 210.6 **ms** (max 284.3) |
| GPU slot wait (back-pressure) | max **220.7 ms** |
| both stages executing at once | **630.6 ms = 39.4 % of NPU busy, 70.5 % of GPU busy** |

Read those last two rows together: publish→consume for chunks 2–4 is 200–280 ms because the
consumer is **mid-block** when the producer publishes (one NPU block, 386 ms, is longer than
one GPU block, 187 ms), so the "latency" is queueing, not handoff cost — the same reading as
P3.3, and the reason P3.1 measured the primitive free-running at 680.9 µs. The producer
blocked 220 ms on slot reuse for exactly the same reason. With a 2-slot ring and a
consumer that is 2× slower per chunk, the producer runs one chunk ahead and then waits: that
is the shape P3.5 has to change by making the GPU stage the long pole (which it is at the
cut's real 160 blocks/chunk).

## 3. Two chunks × 20 NPU blocks — the cut's real stage shape

| quantity | value |
|---|---|
| wall | 25.4 s |
| NPU blocks run | **40** (2 × 20), median **386.6 ms**, 382.9–420.5 |
| NPU chunk window (consume → last block end) | **9.34 s / 9.25 s** |
| … of which the 9-GEMM stage | **7.80 s / 7.73 s** (20 × 386.6 ms) |
| GPU: both chunks published in | **0.214 s** (GPU busy 452.4 ms) |
| both stages executing at once | **212.4 ms = 1.1 % of NPU busy** |

### 3a. Finding: the ring's "chunk" is not the cut's chunk

`proc_a` publishes **one chunk per block**; the cut hands off **once per `k_b` = 160 blocks**
and the NPU then runs 20. So in this run the GPU finished its whole stream in 0.214 s and
idled for 18.4 s while the NPU chewed 40 blocks. The coordinator, the ring and the
back-pressure all behave correctly; the **granularity** is what does not match the paper yet.
Consequence for P3.5: `proc_a` needs `--pub-every N` (publish the activation of block `N`,
`2N`, …) so the GPU stage per chunk is 160 blocks and the overlap fraction becomes meaningful.
Until that exists, any "steady-state period" quoted from this pipeline would be measuring a
per-block handoff, not the cut.

### 3b. Finding: 80 ms of dead time between NPU blocks, and it is the harness's fault

Per-chunk window 9.34 s against 7.73 s of measured GEMM: the difference is **19 gaps of
~80 ms** (68.7–93.8 ms) *between* consecutive blocks, reproducible in both chunks. The gap is
outside the block's own timer, and the only code between the two `TS|npu|block=` stamps is
`fnv(out_hash, UpOut.data(), UpOut.size())` — a byte-at-a-time FNV over `ffn.up`'s output,
4 tiles × 384 × 8960 = **13,762,560 B per block**. At ~5.8 ns/byte that is 80 ms, which
matches the measured gap. It is an integrity artifact of the P3.3 harness, not the NPU stage:
at 20 blocks/chunk it inflates the stage by **1.62 s (17 %)**, and at 160 blocks/chunk it
would cost 12.8 s per chunk. Fix before P3.5: hash every 8th block, or hash a strided sample
of the buffer — Gate 3 parity does not depend on hashing every byte of every block.

## 4. Thermal — both captures start over the gate, and say so

| capture | zones before | gate (battery + 3) | verdict |
|---|---|---|---|
| `…_231417` (4×1) | TPU 27.0 · VIRTUAL-SKIN 23.2 · battery 22.2 | 25.2 °C | **1.8 °C over** — floor-grade |
| `…_231451` (2×20) | TPU 29.0 · VIRTUAL-SKIN 23.4 · battery 22.3 | 25.3 °C | **3.7 °C over** — comparison only |

Both runs are back-to-back on a phone that was at the gate before the first one, so the TPU
zone carries the previous run's heat. The **mechanism** results (pass/fail, ordering,
handshake, back-pressure, the 80 ms gap) are not sensitive to 2 °C; the **rates** are. For a
table-grade rate table, re-run this tool under `tools/npu/cool_rerun.py` (17 min to the
floor, `--gate-wait` alone is not enough at this duty cycle).

## 5. What this proves / does not

- proves: one command brings both stages up in the required order, the GPU stage publishes
  every chunk, the NPU stage consumes and releases every one, both stages run the real INT8
  artifact from one stored copy, and the overlap between them is measured from their own
  `CLOCK_MONOTONIC` stamps rather than assumed
- does not prove: numerical parity (Gate 3, still behind the host FP32 reference), the
  steady-state period at the cut's cadence (needs §3a and §3b first), or the decode — still a
  **cited** 10.640 s stub, not an execution
- measured here, for the record: GPU block under real co-run **187.2 ms** (10 closures) /
  **168.0 ms** (one closure) against the cut's `T_G` = 147.4 ms, and NPU block **385.9 ms**
  against `T_N`'s 9-GEMM 390.4 ms — the import penalty and the closure overhead are the gap,
  both already recorded in P3.2
