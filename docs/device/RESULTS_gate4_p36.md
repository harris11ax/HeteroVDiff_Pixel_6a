# P3.6 — Gate 4: memory and the one-copy proof

*Measured 2026-11-16 00:32 and 00:34 on serial `25311JEGR11811`. Captures
`gate4_20261116_003217` (48 samples) and `gate4_20261116_003411` (54 samples); the
two discovery runs are `gate4_20261116_002554` (smoke, 1 chunk) and
`gate4_20261116_002947` (3 chunks, 68 samples). Tool: `tools/pipe/gate4.py`.*

Gate 4 asks two things of the pipeline while it runs, and neither had been measured:

1. **Proc A + Proc B PSS < 2.2 GiB.**
2. **One mapped copy of the artifact** — proven from `/proc`, not asserted in prose.

Everything below is from the phone. Nothing is modelled.

## Verdict — PASS, twice

| | run `…_003217` | run `…_003411` |
|---|---|---|
| pipeline | GPU published **2/2**, NPU consumed **2/2**, 1440/1440 executes, 0 fatal | same |
| PSS A median | 246,767 kB | 246,873 kB |
| PSS B median | 30,870 kB | 30,907 kB |
| **PSS A+B peak** | **282,580 kB = 0.27 GiB** | **282,751 kB = 0.27 GiB** |
| budget | 2.2 GiB ⇒ **8.2× headroom** | same |
| proc_a artifact mappings | **1** shared `rw-s`, 46,399,488 B, offset 0, inode 314318, in 45/45 samples | 1, 45/45 |
| proc_a zero-copy imports | **10/10** windows | 10/10 |
| proc_b artifact fds | 1, one path only, 54/54 | 1, one path only |
| proc_b artifact mappings | **0**, 54/54 | 0, 54/54 |
| device files the size of the artifact | **1** | 1 |
| `MemAvailable` min | — | 2,380,732 kB |

The earlier 3-chunk run (`…_002947`, 68 samples over 60 GPU-block publishes) measured
the same envelope: PSS A+B peak **282,441 kB**.

## Finding 1 — `smaps` under-reports a GPU process by 2.93×

The same PID, read twice:

| source | Proc A |
|---|---|
| `/proc/<pid>/smaps_rollup` → `Pss` | 85,137 kB |
| `dumpsys meminfo <pid>` → `TOTAL PSS` | 249,290 kB |
| of which the `GL mtrack` row | **164,192 kB** |

`GL mtrack` is the Mali driver's own accounting of the CL buffers the process owns. It
never appears in `/proc/<pid>/smaps`, so a smaps-only reading of a GPU stage is wrong by
**2.93×** (`249,290 / 85,137`). Gate 4 therefore uses the `dumpsys` TOTAL PSS — the
larger, conservative number — and records the smaps figure only as a cross-check.
Proc B has no `GL mtrack` row at all, and there its two sources agree to 0.1 %
(38,381 vs 38,442 kB), which is the control for the effect.

Consequence for the paper: any memory table that quotes `smaps` for the Mali stage
understates it by ~3×, i.e. this is the difference between "0.09 GiB" and "0.24 GiB" per
stage.

## Finding 2 — the device carried a second copy of the block before this run

The first 3-chunk run reported **two** device files of exactly 46,399,488 B:
`hv_blk.bin` (the pipeline's artifact, used by `proc_a`/`proc_b`) and
`hv_block_int8.bin`, dated 14:31 — the T_N harness's own copy of the same block, left
over from `tools/android/npublock/npu_block.cpp`. The pipeline never made that copy, but
its presence falsified the one-copy claim on the device, so the run was recorded as FAIL
and the stale blobs were removed (all regenerable from `artifacts/blocks/`):

| file | bytes | what it was |
|---|---|---|
| `hv_block_int8.bin` | 46,399,488 | the T_N harness's copy of the same block INT8 |
| `hv_down_chunk2/4/7.bin` | 3 × 13,762,560 | `ffn.down` K-slices staged for the TPU branch |
| `hv_w.bin` | 13,762,560 | weight staging blob (probe2) |
| `hv_attn_int8.bin` | 2,498,560 | mid-block attention weights |
| `hv_shared.bin` | 33,554,432 | Gate-0 MAP_SHARED test buffer |

**137,502,720 B (131.1 MiB) freed.** Re-running `npu_block` (T_N table, `ffn.down`,
attention, sustained soak) now requires re-pushing those blobs first — no Python tool in
the tree pushes them.

## What the one-copy proof actually rests on

- **GPU side:** one mapping of the block file, `rw-s` (MAP_SHARED), offset 0, size exactly
  the file, present in every sample; and `proc_a`'s own RESULT line reports
  `imported_weights=10/10`, i.e. every weight window is a `clImportMemoryARM` view of that
  mapping. No window is copied — which is what makes the 161–202 ms block rate possible.
- **NPU side:** `proc_b` opens the block file once, hands the fd to
  `ANeuralNetworksMemory_createFromFd(PROT_READ)`, and never maps it itself (0 regions in
  54/54 samples). The pages belong to the NNAPI HAL, so `proc_b`'s PSS does not contain
  44.25 MiB of weights. The fd count is not the invariant — it measured 1 steady-state and
  4 transiently while the 9 graphs compiled (P3.5a binaries) — the invariant is one path,
  one inode, no mapping.
- **Device:** exactly one file of the artifact's size, so there is no second stored copy
  for either stage to find.

## What this does not prove

- **Numerical parity across the handoff (Gate 3).** `proc_b` hashes the payload it
  consumes, but nothing yet compares either stage's output against an int32 reference for
  the same block.
- **The decode as an execution.** `--decode 1` is still a stub; the 10.554 s TPU decode is
  a cited constant, not work the NPU stage performs.
- **Rate grade.** These runs started 5–13 °C over the thermal gate (TPU 35.0 / 41.0 °C vs
  a 30.2 / 31.3 °C line, back-to-back). Gate 4 is a memory measurement, so the gate is
  recorded and not enforced; the period numbers stay the ones in
  `RESULTS_stream_p35.md`.

## Reproduce

```bash
python tools/pipe/gate4.py --chunks 2 --pub-every 160 --nblocks 20 --poll 1
python tools/pipe/gate4.py --chunks 3 --pub-every 160 --nblocks 20 --poll 1   # longer window
```

Writes `docs/device/gate4_<ts>.{json,md,raw.log,meminfo.txt}`. FAILs if the pipeline does
not publish/consume every chunk, if PSS A+B crosses the budget, if `proc_a` holds more
than one artifact mapping (or a non-shared one), if `proc_a`'s import count is not N/N, if
any `proc_b` fd points somewhere else, or if the device holds more than one file of the
artifact's size.
