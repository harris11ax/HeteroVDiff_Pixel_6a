# Phase 3 — two-process chunk pipeline (build order)

Ordered so that **every tick does one bounded unit** and each unit is verifiable on
its own. Rates are measured (`configs/rates_preset_a.json`); this phase is code.

Target: `python phase3_pipeline/stream.py --chunks 10` runs on the phone, unattended,
GPU process on blocks `[0,160)` and NPU process on `[160,180)` + decode, one handoff
per chunk, and reports the steady-state period to compare against the **23.61 s**
prediction already written down (`RESULTS_sustain_npu.md`, `solve_cut.py`).

## The one invariant to design around

**One INT8 copy of the base weights, mapped by both processes.** Both stages `mmap`
the same file `MAP_SHARED` (route B, validated in Gate 2) and read disjoint weight
regions from it. Nothing about the pipeline is allowed to load, unpack or copy the
1.39 GB artifact: a second copy breaks the paper's premise *and* Gate 4's memory
budget (Proc A + Proc B PSS < 2.2 GiB).

## Handoff = one file, two slots

`/data/local/tmp/hv_handoff.bin`, `mmap(MAP_SHARED)` in both processes:

```
struct SlotHdr {           // 64 B, one per slot
  alignas(8) uint64_t seq;      // chunk index published by the producer, 0 = empty
  alignas(8) uint64_t done;     // chunk index released by the consumer
  uint32_t bytes;               // payload length actually written
  uint32_t flags;               // bit0 = producer finished the stream
  uint64_t t_prod_ns;           // CLOCK_MONOTONIC at publish
  uint64_t t_cons_ns;           // CLOCK_MONOTONIC at consume
};
// payload: slot k at off(k) = sizeof(headers) + k * CHUNK_BYTES
```

- Two slots so the GPU can start chunk `i+1` while the NPU still reads chunk `i`.
  With one slot the stages serialise and the measured cut is not what runs.
- Publishing is `__atomic_store_n(&hdr->seq, i, __ATOMIC_RELEASE)` after the payload
  writes and **before** any wake; the consumer spins with `__ATOMIC_ACQUIRE` and
  `memcpy`s out. On a non-coherent-by-default mapping the `msync(MS_SYNC)` is done
  on the payload range, not the whole file — 4 KiB `msync` per chunk.
- `CLOCK_MONOTONIC` stamps on both sides are what make the artifact prove the
  handoff happened (same evidence style as `RESULTS_corun_tg.md`).

`CHUNK_BYTES` = the residual-stream payload for one chunk at Preset A
(`M = 1530`, `d = 1536`, fp16) = 4,700,160 B = 4.48 MiB; ring = 2 slots ≈ 8.96 MiB.

## Build order (one item per tick)

- ~~**P3.1 — handoff primitive, no compute.** `phase3_pipeline/handoff.h` +
  `tools/android/pipe/handoff_selftest.cpp`~~ — **done 2026-11-15 22:03**,
  `docs/device/RESULTS_handoff_p31.md`: 200 chunks × 2 configurations, **0
  mismatching words, 200/200 in order**, publish→consume **680.9 µs** mean
 (661.7 median, 1023 µs max) free-running and **2541.4 µs** with a 1 ms consumer
  stall; `msync` 4 KiB median **63.7 µs**; ring file exactly **9,408,512 B**.
  Split launch (`--role producer|consumer` from two adb sessions) also passes.
  Producer blocks on 198/200 chunks even free-running — back-pressure is the
  normal path, which is why two slots are required rather than nice to have.
- ~~**P3.2 — GPU stage over real weights.** `tools/android/pipe/proc_a.cpp`: mmap the
  artifact, run the existing tiled INT8 GEMM block (`gemm8.cl`) for k blocks, write
  the residual payload into the slot. Timing per block emitted as `TS|gpu|..`.~~
  — **done 2026-11-15 22:25** → `docs/device/RESULTS_proc_a_p32.md`. Real artifact,
  10/10 weight windows imported zero-copy, 80/80 GEMMs, 8/8 chunks drained by a
  second PID. **Import costs 32 %/block**: 214.5 ms (662 GOPS) zero-copy vs 162.2 ms
  (875 GOPS) with device-local staging, which reproduces the 161.8 ms block rate.
  Attention +4 %, FFN up/down +15/16 % — the penalty tracks the weight working set
  that misses cache. `k_b*` would be 157 (staging) or 148 (import) vs the locked 160:
  **a fork for P3.4 to present, not to apply.**
  Open: payload conversion is host-side (44–57 ms/block) — an fp16 epilogue in
  `gemm8_tiled` removes it before any period number is quoted.
- ~~**P3.3 — NPU stage over real weights.** `tools/android/pipe/proc_b.cpp`: wait on
  `seq`, run `(180 - k)` blocks as the held-compilation FC graphs (the `corun` code
  path), then the decode stub, publish `done`.~~ — **done 2026-11-15 22:53** →
  `docs/device/RESULTS_proc_b_p33.md`. Real artifact via one `createFromFd` view +
  per-GEMM `setOperandValueFromMemory` windows (one stored copy), 9 held graphs
  compiled in 4.52 s, **108/108 executes**, 3/3 chunks consumed, producer/consumer
  payload hash chain **MATCH**. **The 9-GEMM stage measures 381.1 ms/block (263 GOPS)
  — 0.976× the co-run 390.4 ms the cut was solved with, through real weights and the
  ring.** `ffn.down` both branches priced in the same harness: TPU 7×K=1280+6×ADD
  **272.3 ms** vs Mali **47.33 ms** ⇒ **5.75× slower** (whole block 656.8 vs 390.5 ms).
  TPU branch needs a 13,762,560 B weight repack (row-major cannot be K-sliced) —
  **reported as a one-copy deviation, not adopted**. Ring: first-chunk handoff 0.163 µs,
  later chunks queue (consumer mid-block). Traps: word-vs-byte hash made the integrity
  chain unmatchable; a one-tile buffer indexed `tiles×TILE` segfaults the consumer and
  reads as `prod_slot_never_freed`. Thermal: `skip` started 0.2 °C over the floor gate,
  `stage` 4.1 °C over (not table-grade — comparison only).
- ~~**P3.4 — coordinator.** `stream.py`: thermal gate → push → launch both with
  `setsid`, one `adb shell`, collect `TS|` lines, compute per-chunk period, publish→
  consume latency, GPU/NPU busy times, and the overlap fraction. Prove **2 chunks**.~~
  — **done 2026-11-15 23:1x** → `docs/device/RESULTS_stream_p34.md`, captures
  `stream_20261115_231417` (4 chunks × 1 NPU block) and `stream_20261115_231451`
  (2 chunks × 20 = the cut's real stage). Both **PASS**: 4/4 and 2/2 chunks
  published/consumed, 144/144 and 1440/1440 executes, 10/10 zero-copy imports,
  no fatal. Launch order is a **correctness rule**: both binaries
  `reset_headers()` on map, so the consumer must reach `READY` before the
  producer starts (otherwise a publish is wiped and the run hangs 300 s later as
  `chunk_never_published`); that ordering gives a first-chunk handoff of
  **0.244 µs**. Measured: GPU block **187.2 ms** median (10 closures) / **168.0**
  (one closure), NPU block **385.9 ms**, overlap **39.4 % of NPU busy**
  (70.5 % of GPU busy) at 1 block/chunk. Two findings, both P3.5 prerequisites:
  (a) `proc_a` publishes **per block**, the cut hands off once per `k_b` = 160
  blocks → the GPU finished 2×20-block chunks in 0.214 s and idled 18.4 s
  (overlap 1.1 %); needs `--pub-every N`. (b) The P3.3 harness hashes `ffn.up`'s
  13,762,560 B output **every block** (byte-at-a-time FNV) → a reproducible
  **~80 ms gap between blocks**, 1.62 s per 20-block chunk (**17 %** of the
  stage), 12.8 s at 160 blocks/chunk; sample it before quoting any period.
  Both captures start 1.8 °C / 3.7 °C over the gate (back-to-back runs) —
  mechanism-grade, not table-grade; re-run under `cool_rerun.py` for rates.
- **P3.5 — 10 chunks.** Steady-state period vs the 23.61 s prediction; per-chunk
  drift table (the same first-20 %/last-20 % method as the sustained soak).
- ~~**P3.6 — Gate 4.** `dumpsys meminfo` PSS for both PIDs, and `/proc/<pid>/maps`
  compared to prove one mapped copy of the artifact.~~ — **done 2026-11-16 00:34** →
  `docs/device/RESULTS_gate4_p36.md`, `tools/pipe/gate4.py`, captures
  `gate4_20261116_003217` / `…_003411`. **PASS twice**: PSS A+B peak **282.8 MB
  (0.27 GiB) vs the 2.2 GiB budget (8.2× headroom)**; `proc_a` holds **one `rw-s`
  MAP_SHARED view** of the 46,399,488 B block (offset 0) with **10/10** windows
  imported by reference; `proc_b` opens one path, hands the fd to NNAPI, and keeps
  **0 mappings** of it. Two findings: (a) **`smaps` under-reports a GPU process by
  2.93×** — `GL mtrack` 164,192 kB is driver-accounted CL memory and never appears in
  `/proc/<pid>/smaps` (85,137 kB) — so Gate 4 uses the `dumpsys` TOTAL PSS; (b) the
  device carried a **second 46,399,488 B copy** of the same block
  (`hv_block_int8.bin`, left over from the T_N harness) plus 128 MiB of stale weight
  scratch, removed — **137,502,720 B freed**, and `npu_block` must re-push its blobs
  before the next run. Gate 3 (numerical parity) stays parked behind the host FP32
  reference (`.venv-ref`) and is *not* a prerequisite for the period measurement —
  say so explicitly wherever the pipeline numbers are used.
- **P3.7 — the decode as an execution, and Gate 3's parity.** `--decode 1` is still
  a stub: wire the measured 10.554 s TPU decode into the NPU stage as real work, and
  make `proc_b` compare the payload it consumes against an int32 reference for the
  same block instead of only hashing it.
  **P3.7a half done 2026-11-16 01:01** → `docs/device/RESULTS_gate3_parity.md`. The
  comparison exists and is exact (`gate3_parity.py`, `proc_b --dump-parity/--act`), and
  it says the stage **fails**: only **25.8 %** of each output tile matches exact integer
  accumulation — the last 384-column block is bit-exact for every row and row 0 is
  bit-exact for every column, nothing else is. Reference proved by those same regions;
  async-write, alignment, transpose, stride, permutation, shift, stale-GEMM and
  weight-echo hypotheses all ruled out by measurement. **Next: bisect with a
  single-output-tile (`N = 384`) graph.** The decode is still a stub.

## Rules that carry over

- One blocking call < ~8 min; anything longer runs detached with
  `docs/device/.measurement_in_flight` written first.
- One graph per GEMM, M ≤ 384 on the NPU, `adb logcat -c` before any NNAPI work.
- Never more than one timed run at a time: the handoff is a natural mutex, but the
  thermal gate is the real one.
