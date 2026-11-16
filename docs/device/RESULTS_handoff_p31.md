# P3.1 — chunk handoff, two processes, one mapping

*Measured 2026-11-15 22:03:06, serial `25311JEGR11811`, capture
`docs/device/handoff_selftest_20261115_220306.json`. Tool:
`python tools/pipe/handoff_selftest.py --chunks 200`.*

The pipeline's cut only means anything if a chunk produced by the GPU stage can
be handed to the NPU stage **without a copy and without a tear**. This is that
primitive, alone: no compute, no OpenCL, no NNAPI. If it were wrong, every later
number would be a number about the wrong thing.

## What was run

`phase3_pipeline/handoff.h` — one file, two slots, mapped `MAP_SHARED` by both
processes. A 64 B header per slot (`seq`, `done`, `bytes`, `flags`, `t_prod_ns`,
`t_cons_ns`, `payload_off`), headers padded to one page so every payload slot is
page-aligned, payload stride `ceil(4,700,160 / 4096)·4096 = 4,702,208 B`
(Preset A: `M = 1530` tokens × `d = 1536`, fp16 = 4.48 MiB per chunk).

Protocol: producer writes the payload → `msync` → stamps `t_prod_ns` →
`__atomic_store_n(seq, chunk, RELEASE)`. Consumer spins `ACQUIRE`, stamps
`t_cons_ns` immediately after the load, verifies, then
`__atomic_store_n(done, chunk, RELEASE)`. **A slot may not be reused until the
consumer has released it** (`done >= chunk - SLOTS`) — that is what makes a torn
payload impossible to observe rather than merely unlikely.

The producer stamps every 64-bit word with `chunk·0x9E3779B97F4A7C15 ^ j·0x100000001B3`;
the consumer recomputes all 587,520 words per chunk and counts mismatches. A
short payload, a torn payload, or an out-of-order publish all surface as counts.

## Result — PASS, both directions

| run | chunks | bad words | out of order | publish→consume (mean / median / max) | producer blocked |
|---|---|---|---|---|---|
| free-running (`--stall-us 0`) | 200 | **0** | 0 | **680.9 / 661.7 / 1023.4 µs** | 198/200, median 621 µs |
| back-pressured (`--stall-us 1000`) | 200 | **0** | 0 | **2541.4 / 2594.4 / 8167.2 µs** | 198/200, median 1551 µs |

- every chunk index arrived exactly once, in order, on both sides (200/200)
- `t_cons > t_prod` on 400/400 publishes — no reordered clock read
- device ring file **9,408,512 B = 4096 + 2 × 4,702,208 B** — exactly two slots;
  a third buffer would have shown up here as a size mismatch
- `msync(4 KiB, MS_SYNC)` per chunk: **median 63.7 µs**, mean 88.9 µs, worst
  2.13 ms. At a 23.6 s chunk this is 0.0004 % of the period, so the pessimistic
  barrier is affordable and there is no reason to relax it to `MS_ASYNC`.
- thermal, recorded not gated (correctness test, the ring lives in page cache):
  TPU 25.0 → 25.0 °C, battery 21.8 → 21.8 °C

Two processes, two PIDs, one mapping: fork mode reported producer pid 7560 /
consumer pid 7561 in the smoke run, and a *split* launch from two separate `adb
shell` sessions with `--role producer` (setsid, detached) / `--role consumer` also
passed 30/30 chunks with `stream_done_flag=1` — that is the shape P3.2/P3.3 will
use, so it is proven now rather than discovered later.

## Two things this measurement says that were not obvious

1. **Back-pressure is the normal path, not an edge case.** Even free-running, the
   producer blocked on 198 of 200 chunks: writing 4.7 MB costs ~0.15 ms while the
   consumer's verify pass costs ~0.6 ms. The published→consumed latency is
   therefore *the consumer's read time*, not the protocol's — the store+spin cost
   is under the 82 µs floor seen in the first chunk of each run. A one-slot ring
   would have serialised this test into 200 × (write + read); the second slot is
   what lets the producer run ahead, and the `done` gate is what stops it running
   too far ahead.
2. **The 4 KiB `msync` is cheap enough to keep.** Median 63.7 µs per chunk. The
   plan specified it as a coherency/ordering belt-and-braces measure; it is now a
   measured number rather than an assumed cost.

## What this does *not* establish

- No period, no overlap, no CPU-time split — those need the real stages (P3.4,
  P3.5) with the 23.61 s prediction in hand.
- Nothing about the INT8 weight artifact: the ring carries **residual
  activations only** and holds zero weight bytes, so the one-copy rule (Gate 4)
  is untouched here by construction — Gate 4 still has to prove the *weight*
  mapping is single, later, with `/proc/<pid>/maps` on both stages.
- Single-chunk-per-slot reuse is proven at 200 chunks with a 1 ms consumer stall;
  if the real NPU stage ever stalls a whole chunk (>23 s) the ring's behaviour is
  the same by construction but is not measured — P3.4/P3.5 will show it.

## Traps paid for in this unit

- **Two processes on one stdout pipe corrupt the log.** Block-buffered stdio
  flushes at arbitrary byte offsets, so the producer's line and the consumer's
  line merged mid-record and the timestamps became unparseable — the one thing
  the test exists to prove. Fix: `setvbuf(stdout, NULL, _IOLBF, 0)` in every
  process, plus a `malformed_lines` counter in the host parser so a recurrence
  fails the verdict instead of being silently dropped.
- **`#include "../../phase3_pipeline/handoff.h"` from `tools/android/pipe/`
  needs three `..`**, not two (`tools/android/pipe` → repo root), else CMake's
  `-I` points at `tools/phase3_pipeline`.
- The producer — not the consumer — owns ring creation and the header reset, and
  only when the file was absent/empty. A consumer that reset would wipe a
  publish already in flight.
