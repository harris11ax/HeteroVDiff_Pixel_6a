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
- **P3.2 — GPU stage over real weights.** `tools/android/pipe/proc_a.cpp`: mmap the
  artifact, run the existing tiled INT8 GEMM block (`gemm8.cl`) for k blocks, write
  the residual payload into the slot. Timing per block emitted as `TS|gpu|..`.
- **P3.3 — NPU stage over real weights.** `tools/android/pipe/proc_b.cpp`: wait on
  `seq`, run `(180 - k)` blocks as the held-compilation FC graphs (the `corun` code
  path), then the decode stub, publish `done`.
- **P3.4 — coordinator.** `stream.py`: thermal gate → push → launch both with
  `setsid`, one `adb shell`, collect `TS|` lines, compute per-chunk period, publish→
  consume latency, GPU/NPU busy times, and the overlap fraction. Prove **2 chunks**.
- **P3.5 — 10 chunks.** Steady-state period vs the 23.61 s prediction; per-chunk
  drift table (the same first-20 %/last-20 % method as the sustained soak).
- **P3.6 — Gate 4.** `dumpsys meminfo` PSS for both PIDs, and `/proc/<pid>/maps`
  compared to prove one mapped copy of the artifact. Gate 3 (numerical parity) stays
  parked behind the host FP32 reference (`.venv-ref`) and is *not* a prerequisite for
  the period measurement — say so explicitly wherever the pipeline numbers are used.

## Rules that carry over

- One blocking call < ~8 min; anything longer runs detached with
  `docs/device/.measurement_in_flight` written first.
- One graph per GEMM, M ≤ 384 on the NPU, `adb logcat -c` before any NNAPI work.
- Never more than one timed run at a time: the handoff is a natural mutex, but the
  thermal gate is the real one.
