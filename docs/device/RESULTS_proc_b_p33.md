# P3.3 — the NPU stage over the real INT8 artifact

*Measured 2026-11-15 22:47, 22:53 and 22:54 on serial `25311JEGR11811`. Reproduce with
`python tools/pipe/proc_b.py --chunks 3 --down skip` and `… --down stage`. Raw
captures: `proc_b_20261115_225443.*` (skip, **both quoted figures below come from this
run** — same binary as the stage run), `proc_b_20261115_225321.*` (stage), and
`proc_b_20261115_224720.*` (skip, measured before the `UpOut` buffer fix: 381.1 ms, i.e.
inside the same 2 % band). Device-side: `tools/android/pipe/proc_b.cpp`; host driver
`tools/pipe/proc_b.py`.*

The pipeline's NPU half now exists and runs against the **real** artifact: it is the
consumer side of the P3.1 ring, `createFromFd`s `artifacts/blocks/block_00.bin` once and
binds each of the nine compilable GEMMs' weight window with
`setOperandValueFromMemory` (offsets, no copy), holds all nine compilations live, waits on
`seq`, runs `(180 - k)` blocks as 4 token tiles of 384, then releases the slot with `done`.

## Verdict

**PASS on both `ffn.down` branches** — 108/108 held-graph executes per run, 3/3 chunks
consumed and released, payload hash chain **MATCH** between the producer and the consumer,
0 non-finite words. The nine-graph compile costs **4.52 s** once per process.

## The number that matters: the 9-GEMM stage reproduces the co-run rate through real weights

| path | 9-GEMM block | GOPS | source |
|---|---|---|---|
| P3.3 NPU stage, real artifact, over the ring (`skip`) | **377.8 ms** | **265** | this run |
| P3.3 NPU stage, second run (`stage`) | 373.4 ms | 269 | this run |
| P3.3 NPU stage, pre-fix binary | 381.1 ms | 263 | 22:47 run |
| co-run harness, synthetic blob (`T_N` input to the cut) | 390.4 ms | 257 | `RESULTS_corun_tg.md` |
| solo per-GEMM table, cool start | 417.8 ms | 222 | `RESULTS_npu_tn.md` `…_160146` |

Three runs of this stage land in **373.4–381.1 ms** (2.1 % spread; 377.8 ms on the run
quoted, 381.1 ms on the pre-fix one), i.e. **0.967×** the co-run figure and **0.904×** the solo table — the stage rate
the cut was solved with is reproduced end-to-end (real bytes, ring handoff, held
compilations, 4 dispatches per GEMM). Per-GEMM, the stage is
`attn1.to_q` 29.8 / `…to_k` 29.3 / `…to_v` 30.0 / `…to_out` 30.2 / `attn2.to_q` 30.5 /
`…to_k` 30.5 / `…to_v` 30.2 / `…to_out` 30.2 / `ffn.up` **144.6** ms per block (4 tiles),
against the solo table's 32.2–35.5 and 146.5 ms. There is no thermal trend across the
three starts (25.0 → 27.0 → 30.0 °C gave 381.1 → 377.8 → 373.4 ms): the ordering is
inverted relative to temperature, i.e. the spread is run-to-run, not throttling.

## `ffn.down` — both branches, measured in the same harness

| path | ms per block | vs Mali | state |
|---|---|---|---|
| Mali-G78 tiled INT8 | **47.33** | 1.00× | measured, `RESULTS_gemm_mali.md` |
| Edge TPU, 7×K=1280 + 6×ADD over the real bytes | **272.3** | **5.75× slower** | **measured here** (`--down stage`) |
| Edge TPU, single FC K=8960 | — | — | structurally impossible (compiler K ceiling 1664) |

So a whole NPU block is either **388.1 ms** (nine GEMMs, down on Mali) or **656.8 ms**
(nine GEMMs + down on the TPU) — the median of 3 blocks in each run. The TPU branch pays
13 dispatches per token tile (7 FC + 6 ADD), exactly as `RESULTS_npu_tn.md` §4 predicted;
its 272.3 ms is +5.6 % over the 258.0 ms measured there on an LCG blob, which is the
expected cost of the real weights' value-dependent gather plus the activation re-layout
(2.2 ms/block).

Two extra costs the TPU branch carries, both measured:

- **weight gather 3.3 ms once** for 13,762,560 B — `[1536,8960]` row-major cannot be
  sliced along K into a legal constant operand (chunk `c` is columns `c*1280..`, stride
  8960), so the slices must be repacked. ⚠️ That repack is a **second stored copy of the
  down weights** for as long as the models live. It is reported as a one-copy deviation
  (the binary prints `WARN|down_staging_copy_bytes=13762560|…`), **not adopted** —
  `configs/` and `design_locks.json` are untouched.
- **down compile 2602 ms** (7 × K=1280 at 340–377 ms, ADD at 98 ms) on top of the 4.52 s
  for the block's nine graphs.

## Other measured numbers (this run)

| quantity | value |
|---|---|
| ring setup | `file_bytes=9,408,512`, 2 slots, stride 4,702,208, chunk 4,700,160 B |
| payload fp16→INT8 quantisation (1530×1536 = 2,349,480 values, `SA = 1/127`) | **6.6 ms** median on the quoted run (5.3–13.8 ms across the three) |
| publish→consume, first chunk | **0.163 µs** when the consumer is already parked (reproduces P3.2's 0.239 µs); 7.9 ms when it is still setting up its buffers |
| publish→consume, later chunks | 391–710 **ms** — this is **queueing**, not handoff cost: the consumer is mid-block when the producer publishes |
| zero / NaN payload words | 31–45 zero words of 2,349,480, **0 NaN** |
| TPU thermal zones | 27.0 → 33.0 °C (quoted `skip`); 30.0 → 36.0 °C (`stage`) |

## Thermal honesty

| run | start (TPU / skin1 / skin2 / VR-skin / battery) | gate (battery+3) | verdict |
|---|---|---|---|
| `skip` 22:47 (pre-fix) | 25.0 / 23.2 / 23.6 / 23.0 / 21.8 °C | 24.8 °C | **0.2 °C over — the idle floor** |
| `stage` 22:53 | 30.0 / 26.3 / 27.4 / 25.5 / 22.9 °C | 25.9 °C | **4.1 °C over — not table-grade** |
| `skip` 22:54 (quoted) | 27.0 / 25.1 / 25.3 / 24.7 / 23.0 °C | 26.0 °C | **1.0 °C over — floor-grade, stated** |

The 22:47 `skip` run sat on the same floor as the table-grade `T_N` capture (25.0 vs a 24.8
gate; this device's TPU zone cannot report below 25.0). The two later runs were taken back
to back with **no cooldown** — 1.0 °C and 4.1 °C over the line — so **the `stage` branch's
down-projection number stands as a comparison (two branches, same harness, same artifact),
not as a table-grade latency**, and the 2 % spread across the three 9-GEMM figures is
run-to-run, not thermal (it moves the *wrong* way with temperature).
Nothing was dropped, and nothing is presented as cool when it was not.

## Three defects this pass paid for

1. **The integrity chain can never match if the two sides hash differently.** The consumer
   hashed 16-bit words (`h = (h ^ v) * prime`), the producer hashed bytes; the chains
   disagreed on a run that was otherwise clean. Both sides now use byte-wise FNV-1a over
   `CHUNK_BYTES`, and the two `SUMMARY` lines are compared by the host driver.
2. **A one-tile buffer indexed as `tiles × TILE` segfaults the consumer** — and from the
   producer's side the symptom is `FATAL|prod_slot_never_freed|chunk=3`, which reads like a
   producer bug. The real cause was the consumer dying before its first `release`. `ffn.up`
   now keeps a per-tile output buffer (`UpOut`, `tiles×TILE×8960`) because `ffn.down`
   consumes *that tile's* output; every other GEMM writes into one scratch tile.
3. **The host driver crashed while formatting the report, after the device run.** The run's
   output now lands in `docs/device/proc_b_<ts>.raw.log` (gitignored) before any parsing, so
   a report-formatting bug can no longer cost a device run.

## What this proves / does not

- proves: the NPU stage consumes what the GPU stage publishes, at the measured stage rate,
  over the real artifact, with **one stored INT8 copy** on the `skip` branch; and that
  `ffn.down` on the TPU costs 5.75× the GPU's time in this harness too.
- does not prove: numerical parity (the activation is the residual payload quantised at
  `SA = 1/127`; the attention/softmax/AdaLN path and Gate 3 parity are untouched), that the
  NPU's per-tensor scales reproduce the locked per-column quantisation (cost measured in
  `RESULTS_scales.md`, decision still the student's), the decode (stub: `T_V` = 10.640 s
  cited from `RESULTS_vae_tv.md`), or the steady-state period (P3.5).
- the two stages have **not yet run as two processes at once** — that is P3.4, and the
  23.61 s period remains a prediction until then.
