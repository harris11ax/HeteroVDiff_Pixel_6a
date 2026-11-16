# P3.2 — the GPU stage over the real INT8 artifact

*Measured 2026-11-15 22:24/22:25 on serial `25311JEGR11811`, two runs back to back.
Reproduce with `python tools/pipe/proc_a.py --blocks 8` (route `import`) and
`… --route copy`. Raw captures: `proc_a_20261115_222445.json` (import),
`proc_a_20261115_222504.json` (copy).*

The pipeline's GPU half now exists and runs against the **real** artifact, not a
synthetic buffer: it `mmap`s `artifacts/blocks/block_00.bin` (44.25 MiB, the exact
bytes Gate 1 verified), imports each of the 10 GEMMs' weight windows into OpenCL,
runs the locked block table with the existing tiled `gemm8.cl` kernel, and publishes
one fp16 residual chunk per block into the 2-slot ring from P3.1.

## Verdict

**PASS** on both routes — 80/80 GEMM dispatches, 8/8 blocks published and drained by
a second PID, 0 non-finite words on the consumer side, publish/consume latency
0.24 µs mean. **10/10 weight buffers imported zero-copy** on the `import` route.

## The finding: zero-copy costs the GPU 32 % per block

Same binary, same weights, same thermal conditions (second run started *hotter*),
only the weight path differing:

| route | block (10 GEMMs, one closure) | GOPS | vs co-run 0.1474 s |
|---|---|---|---|
| `import` — `clImportMemoryARM(HOST_ARM)` over the mmap (**one stored copy**) | **214.5 ms** median (min 198.0) | 662 | 1.455× |
| `copy` — per-GEMM `CL_MEM_COPY_HOST_PTR` into device memory | **162.2 ms** median (min 153.8) | 875 | 1.100× |

The copy route's 162.2 ms independently reproduces the **161.8 ms** block rate from
`RESULTS_gemm_mali.md`, which is the check that this harness measures the same thing.

The penalty is not uniform, and where it lands says what it is:

| GEMM | shape | `import` ms | `copy` ms | delta |
|---|---|---|---|---|
| `attn1.to_q` … `attn2.to_out` (×8) | `[1536,1536]` | 10.18 – 11.08 | 9.56 – 10.33 | +4 % |
| `ffn.net.0.proj` (up) | `[8960,1536]` | 55.21 | 47.84 | **+15 %** |
| `ffn.net.2` (down) | `[1536,8960]` | 56.39 | 48.41 | **+16 %** |

The attention weights (2.25 MiB each) still fit close to the GPU's cache and pay
almost nothing; the two FFN weights (**13.8 MiB each**, 59 % of the block's MACs)
stream from host memory every block and pay 15–16 %. So the cost of route B is
proportional to the weight working set that misses cache, not a fixed import tax.

## What this does to the cut (reported, not applied)

`phase2_profile/solve_cut.py`, `T_N = 0.6484 s`, `T_V = 10.640 s`, `B = 180`:

| `T_G` source | `k_b*` | period | speedup vs NPU-only |
|---|---|---|---|
| co-run harness, device-local weights (current lock) | **160** | 23.61 s | 5.39× |
| P3.2 `copy` route, 162.2 ms | 157 | 25.55 s | 4.98× |
| P3.2 `import` route, 214.5 ms | 148 | 31.75 s | 4.01× |

`configs/rates_preset_a.json` and `design_locks.json` are **untouched**. This is a
fork for the student, and it is now a comparison:

- **staging copy** — if the GPU copies the *current* block's 44.25 MiB into device
  memory, the on-disk artifact stays the single stored INT8 copy, but 44.25 MiB is
  transiently duplicated in GPU memory. 162.2 ms/block; `k_b*` = 157.
- **route B import** — the bytes are never duplicated anywhere; the stage reads the
  page cache directly. 214.5 ms/block; `k_b*` = 148.

Both branches keep "one stored INT8 copy" as written (a staging copy is a copy of
*one block*, not a second artifact) — which is exactly why this is a decision about
Gate-4 memory budget and per-block rate rather than about the lock's wording.

## Other measured numbers

| quantity | value |
|---|---|
| blocking-read closure (completion barrier) | **1.7 µs**, 20 reads — so the per-GEMM timings are not closure-dominated (this corrected an earlier guess of ~3 ms) |
| payload conversion (read back the `ffn.net.2` output 9.4 MB fp32, convert 1530×1536 to fp16, 4,700,160 B into the slot) | **44.0 ms** median (`copy`), 57.2 ms (`import`) |
| `msync(MS_SYNC)` first page per chunk | 4.2 ms worst-case for the run |
| publish → consumer observe | **0.239 µs** mean, 0.285 µs max (consumer already spinning) |
| sustained drift, `import` series | 198.0 → 219.4 ms over 8 blocks (**+10 %**), TPU zone 25.0 → 34.0 °C |
| sustained drift, `copy` series | 153.8 → 161.8 ms over 8 blocks (**+5 %**), TPU zone 26.0 → 36.0 °C |

## Thermal

| run | before (TPU / skin1 / skin2 / VR-skin / battery) | after (TPU / battery) | gate |
|---|---|---|---|
| import | 25.0 / 22.9 / 23.2 / 22.7 / 21.7 °C | 34.0 / 21.9 °C | start **0.3 °C over** the battery+3 line |
| copy | 26.0 / 23.0 / 23.4 / 23.1 / 21.9 °C | 36.0 / 22.0 °C | start **0.9 °C over** |

Both runs started on the **idle floor** — 25.0 °C is the lowest the TPU zone reports
on this device (it sat there for hours of preflight ticks), and the battery-derived
gate of 24.7 °C cannot be met while the device idles at 25.0. Per the standing rule
these are recorded as floor-grade with the margin stated, not dropped. The *relative*
comparison between routes is unaffected: the copy run began 1.0 °C hotter and was
still 52 ms faster.

## Two defects this pass paid for

1. **`clImportMemoryARM` of a read-only mapping returns `CL_OUT_OF_HOST_MEMORY` (-6)**
   on this driver. Route B works only with an `O_RDWR` file and a
   `PROT_READ|PROT_WRITE` `MAP_SHARED` mapping — the same combination `probe2.cpp`
   used in Gate 2. Nothing writes a byte; the mapping mode is what the importer wants.
2. The first draft timed each GEMM and closed it with a blocking read, then compared
   the block against the 0.1474 s reference and read the 34 ms gap as closure
   overhead. The measured closure is **1.7 µs**. The `once` pass added here (10
   dispatches, one closure) is the form the reference was measured in, and with it
   the gap resolves into the weight-path cost above.

## Still open

- payload conversion is **host-side** and costs 44–57 ms/block (27 % of the staging
  route's block time). A `gemm8_tiled` fp16 epilogue writing `half` directly into the
  slot removes it; until then the *stage* time is compute + conversion, and the
  `T_G` values fed to the cut above are compute-only.
- the residual is the `ffn.net.2` output only — the block's attention/softmax and
  AdaLN paths are not in this stage yet (they are the Gate-3 parity work).
- activations here are a deterministic int8 fill, not a real forward pass: this
  document measures *rates and the handoff*, never image quality.
