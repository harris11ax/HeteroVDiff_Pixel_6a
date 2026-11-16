# P3.5 — ten chunks end to end, and the GPU's sustained-rate knee

*2026-11-16 00:04. Tool `phase3_pipeline/stream.py` for the run, `tools/npu/cool_rerun.py
--cmd` for the gate. Capture `stream_20261116_000415` (JSON + raw log + this run's generated
`stream_20261116_000415.md`). Start was **table-grade**: TPU 25.0 °C, VIRTUAL-SKIN 23.4 °C,
skin 23.6 °C, battery 22.5 °C against a 25.5 °C gate.*

    python tools/npu/cool_rerun.py --max-wait 1200 --cmd \
      "python phase3_pipeline/stream.py --chunks 10 --nblocks 20 --pub-every 160 \
             --hash-every 20 --timeout 900"

Ten chunks is the deliverable's own bar ("streams ≥ 10 chunks on the phone"), and it is the
first run long enough to answer two things P3.5a could only bound: the **steady-state** period,
and whether the 202 ms GPU block holds for a whole trajectory rather than one chunk.

## 1. Verdict

**PASS.** GPU published **10/10** chunks over **1600** blocks, NPU consumed **10/10**,
**7200/7200** held-graph executes (10 × 20 × 9 GEMMs × 4 tiles), 10/10 weight windows imported
zero-copy, one `createFromFd` view on the NPU side (one stored INT8 copy), NaN words **0**,
fatal **0**, `proto_err` 0, wall **370.4 s**. Consumer-first launch order held: first handoff
**0.24 µs**.

| quantity | value |
|---|---|
| GPU block, 10 per-GEMM closures (`import`) | **209.8 ms** median over **n=1600** (168.7–303.4) |
| GPU block, healthy regime (blocks 0–1100) | **203.0 ms** median (range of 100-block bins 199.9–204.7) |
| GPU block, derated regime (blocks 1200–1600) | **274.2 ms** median (**+35.1 %**; first-20 %/last-20 % means +32.4 %) |
| GPU block, one-closure reference (post-stream, n=4) | 251.2 ms, 561.1 GOPS, closure overhead 1.8 µs |
| NPU block on the TPU (9 GEMMs, 4 tiles) | **395.6 ms** median (n=200), **flat** end to end |
| NPU chunk window (quant + 20 blocks) | **7.93 s** median (7.88–7.98, n=10) |
| GPU publish interval, healthy chunks 2–7 | **32.72 s** median (32.61–32.81) |
| GPU publish interval, derated chunks 9–10 | **43.56 / 43.90 s** |
| publish→consume | 0.51 µs mean, **1.14 µs max** |
| GPU slot wait (back-pressure) | **2 µs** max — the ring never binds at this shape |
| payload conversion (host-side fp16) | 50.8 ms/chunk median, 52.9 max |
| `msync` (first page) | 6.6 ms max |
| activation quantisation, NPU side | 5.23 ms/chunk median |
| `ffn.up` output hash | 777.9 ms over 10 calls = **77.8 ms each** (sampled every 20th block) |
| NPU graph compile (9 one-op graphs) | 4584.2 ms, once |
| both engines executing at once | **71.40 s = 90.0 % of NPU busy** |
| wall | 370.4 s |

## 2. The steady-state period is 32.7 s, and the overlap is already at its ceiling

Chunks 2–7 publish every **32.61–32.81 s** (median 32.72) — i.e. the 2-chunk 32.544 s in
P3.5a was **not** a startup artifact. The pipeline therefore holds a **32.72 s** chunk period
in its healthy regime at the locked `k_b = 160`, which is **1.39×** the 23.61 s that
`configs/rates_preset_a.json` predicts from the staging co-run pair.

**The overlap is maximal, which retires the P3.5a caveat.** P3.5a measured 50.2 % of NPU busy
and said 2 chunks structurally understate it. For `N` chunks the ceiling is `(N-1)/N` — the
last chunk's NPU work has nothing left to overlap — so at N=10 the ceiling is **90.0 %**. This
run measures **90.02 %** (71.40 s of 79.32 s). The pipeline cannot overlap more without a
different cut; what is left is not a coordinator defect but the GPU being the long pole by
**4.13×** per chunk (32.7 s of GPU work against 7.93 s of NPU work). The NPU genuinely idles
~76 % of every period at `k_b = 160`.

## 3. New: the Mali stage derates 35 % after ~232 s, and this is the first run long enough to see it

The block time is flat for the first ~1100 blocks and then steps up, permanently:

| blocks | median ms | vs healthy |
|---|---|---|
| 0–100 | 204.6 | +0.2 % |
| 300–400 | 203.2 | −0.4 % |
| 700–800 | 203.5 | −0.3 % |
| 900–1000 | 202.2 | −0.9 % |
| 1100–1200 | 220.2 | +7.9 % |
| 1200–1300 | 269.5 | **+32.0 %** |
| 1300–1400 | 273.8 | **+34.2 %** |
| 1500–1600 | 280.8 | **+37.6 %** |

The 50-block rolling median crosses 110 % of the healthy rate at **block 1135**, i.e. **232 s
of continuous GPU work in** (healthy-rate base 204.1 ms from the first 320 blocks). It is a
**step, not a drift**: −0.9 % over the first 1000 blocks, **+35.1 %** over blocks 1200–1600
(block-bin medians 269.5–280.8 ms), no recovery. The NPU is unaffected in the same window
(396.5 ms first-40 blocks → 394.0 last-40),
so this is not a ring, host or back-pressure effect (slot wait 2 µs max, payload 50.8 ms/chunk
flat).

**Corroboration from the same capture:** the one-closure reference pass runs *after* the stream
(its `TSC|gpu|` lines are the last 4 in the log) and costs **251.2 ms/block** — 1.51× the
**166.2 ms** that the identical route measured in P3.5a from a cold start. So the derate is
engine-wide, not an artifact of the 10-closure timing form.

**Cause is not established, and is not claimed.** DVFS or a thermal trip point are the
candidates; nothing on this phone exposes the GPU frequency without root. What is established is
the shape (a step at 232 s), its stability (−0.9 % for 1000 blocks), and its one-sidedness (GPU
only). Zones over the run: TPU 25.0 → **52.0 °C**, skin_therm1 23.6 → 41.9 °C, battery
22.5 → 28.0 °C, VIRTUAL-SKIN 23.4 → 38.1 °C.

**Why P3.5a missed it:** 320 blocks ≈ 64.7 s of GPU work never reaches the knee. Every number
in `RESULTS_stream_p35a.md` is inside the healthy regime; the claim there that the GPU stream
"shows no throttle term" holds only up to 232 s. A 180-block trajectory at any `k_b ≥ 128`
runs the GPU stage past it.

## 4. Cut consequence — measured, reported, not applied

`T_G` is a short-run rate. The pipeline's measured periods:

| configuration | GPU ms/block | period | vs 23.61 s prediction |
|---|---|---|---|
| staging co-run (`k_b*` input, locked) | 147.4 | 23.61 s | 1.00× |
| import, one closure (healthy) | 166.2 | 26.59 s | 1.13× |
| import, 10 closures (healthy) | 203.0 | **32.72 s measured** | 1.39× |
| import, 10 closures (derated) | 274.2 | **43.90 s measured** | 1.86× |
| import, 10 closures (**run average**, all 1600 blocks) | 222.2 | 35.6 s | 1.51× |

`solve_cut.py` with `T_N` = 0.6484, `T_V` = 10.640, `B` = 180:

| `T_G` (s) | what it is | `k_b*` | period | speedup vs NPU-only |
|---|---|---|---|---|
| 0.1474 | staging co-run — locked | 160 | 23.61 s | 5.394× |
| 0.1662 | import, one closure | 157 | 26.09 s | 4.881× |
| 0.2021 | import, 10 closures, healthy | 150 | 30.32 s | 4.201× |
| **0.2742** | **import, 10 closures, derated** | **138** | **37.87 s** | 3.363× |

The locked `k_b* = 160` is a property of a rate the pipeline cannot sustain: the honest
as-built period is 32.7 s healthy and 43.9 s once the GPU derates, against 23.61 s predicted.
`configs/rates_preset_a.json` and `configs/design_locks.json` are **untouched** — this is a
decision for the student, with numbers on both sides. The one lever this run did *not* measure
in the derated regime is the one-closure or staging form (only a 4-block post-stream sample at
251.2 ms exists), so "how much of the 33 % the closure form is responsible for" stays open.

## 5. Thermal and grade

- before: battery 22.5, TPU 25.0, VIRTUAL-SKIN 23.4, skin_therm1 23.6, skin_therm2 23.9
- after: battery 28.0, TPU 52.0, VIRTUAL-SKIN 38.1, skin_therm1 41.9, skin_therm2 41.5
- gate (max zone ≤ battery + 3): **PASS** at the start (25.0 vs 25.5 °C) after a 90 s wait on
  `cool_rerun.py` (`.cool_rerun.log`, 23:57:54 PASS, confirm sample 23:58:04 PASS)

So the **healthy-regime** rates (§1 first two rows, §2 periods) are table-grade. The derated
rows are a *different measured configuration*, not a degraded version of the same one: they are
what the hardware does after 232 s of that load, and quoting them as the pipeline's period
under a 180-block trajectory is the correct use.

## 6. What this proves / does not

- proves: the two-process pipeline runs **10 chunks / 1600 GPU blocks / 200 NPU blocks** end to
  end on the phone with no manual intervention, one handoff per chunk, handoff latency
  **≤ 1.14 µs**, no back-pressure, zero-copy on both sides, and the deliverable's "≥ 10 chunks"
  bar is met at the cut's real 160/20 shape
- proves: steady-state overlap is **at its structural ceiling** (90.02 % measured vs 90.0 %
  for N=10), so the residual period is the GPU stage, not coordination
- proves: the GPU stage **derates +35.1 %** at block ~1135 (232 s), step-shaped, NPU unaffected
  — a term absent from every rate table built from runs shorter than 4 minutes
- does **not** prove: numerical parity (Gate 3 — still unbuilt), the decode (still the cited
  10.640 s stub, `--decode 1` is a no-op in this harness), or the cause of the derate
- does **not** prove: that the derate is thermal rather than a governor policy — no frequency
  counter is readable, so DVFS stays a candidate
- note: payloads are byte-identical run to run (`140442ac266bc92b`, 6 zero words/chunk, 0 NaN)
  because `proc_a` runs fixed activation buffers; the NPU's hash chain
  (`76798d892b7e0ed3` over 10 payloads) is a per-chunk integrity record, not a parity check
