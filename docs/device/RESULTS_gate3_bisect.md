# Gate 3 bisect — WHERE the `FULLY_CONNECTED` tile breaks (P3.7b) — 2026-11-16

**Status: LOCALIZED.** Gate 3 measured a `[384, N]` tile that was only **25.8 %** bit-exact and
showed the correct region was structural — the last 384-column block for every row, row 0 for
every column. The bisect says which half of that signature is the defect: **the M (row) direction**.
A 384-column output block is **bit-exact, 0 LSB of error, on every element**, at `M ≤ 200`; the same
block at `M = 384` **does not even compile**. Wide graphs (`N = 1536`, `N = 8960`) at `M = 256…384`
are the shapes that return 25.8–50.7 %.

Tools: `tools/pipe/gate3_bisect.py` (host verifier) + `proc_b --probe-dir DIR --probe-g G
[--probe-plan "M,N,col0;…"]` (device probe; `proc_b.cpp` §"Gate 3 bisect"). Captures
`gate3_bisect_20261116_012157` … `_012349` (5 runs, 32 probe graphs, all on `block_00.bin`; the
`.json` / `.raw.log` siblings are gitignored, `gate3_bisect_20261116_012157.md` is the full
discriminating table).

## Method (unchanged from Gate 3, so the numbers are comparable)

One `FULLY_CONNECTED` per probe, per-tensor scales `SA = SW = 1/127`, zero bias, weight operand a
window into the **one** file-mapped INT8 artifact at `off + col0·K` (no copy, still route B).
Activations are the `basis` pattern `A[r][r % K] = 127`, so the exact output element is
`W[col0+n][r % K]` — one stored weight byte per element, no saturation. The host recomputes
`acc = Σ a·w` exactly (fp32 BLAS, K-split partials < 2²⁴, int64 accumulate) and applies NNAPI's
requant `clamp(round(acc/127))`. "bit-exact" below = fraction of the tile with **Δ = 0 LSB**.

## Measured — the shape map (`attn1.to_q`, N_out 1536, K 1536, real INT8 weight bytes)

| M \ N | 128 | 384 | 768 | 1536 | 8960 |
|---|---|---|---|---|---|
| **1** | — | **100.00 %** | — | **100.00 %** | — |
| **8** | — | **100.00 %** | — | — | — |
| **128** | **100.00 %** | **100.00 %** | **100.00 %** | compile **OP_FAILED** | — |
| **160** | — | **100.00 %** | — | — | — |
| **192** | **100.00 %** | **100.00 %** (×2 runs) | compile **OP_FAILED** | compile **OP_FAILED** | 14.16 % |
| **200** | **100.00 %** | **100.00 %** | — | — | — |
| **208** | — | compile **OP_FAILED** | — | — | — |
| **216** | — | compile **OP_FAILED** | — | — | — |
| **224** | — | compile **OP_FAILED** | — | — | — |
| **256** | — | compile **OP_FAILED** | compile **OP_FAILED** | 50.67 % | — |
| **320** | — | compile **OP_FAILED** | — | 50.60 % | — |
| **384** | — | compile **OP_FAILED** (×3 runs) | compile **OP_FAILED** | **25.83 %** | (Gate 3: 4.54 %) |

`ffn.up` (N_out 8960, K 1536, 13.7 MB of the *same* weight bytes) obeys the **same boundary**:
`M=128/192/200 × N=384` → **100.00 %**, `M=192 × N=768` → `OP_FAILED`, `M=256 × N=384` →
`OP_FAILED`, `M=192 × N=8960` → 14.16 %. So the rule is **shape-driven, not weight-driven**, which is
what makes it a compiler property rather than a data property.

The `M=384, N=1536` control reproduces Gate 3 to two decimals (25.828 % both runs) — the harness is
the same measurement, not a new one.

### Where the correct elements are, per 384-column block (measured, wide graphs)

| probe | c0 | c384 | c768 | c1152 | row 0 | row 1 | row M−1 |
|---|---|---|---|---|---|---|---|
| `M=384 N=1536` | 1.12 % | 1.11 % | 1.09 % | **100.00 %** | 100 % | 25.33 % | 25.39 % (row 383) |
| `M=256 N=1536` | 50.69 % | 50.68 % | 50.65 % | 50.68 % | 100 % | 50.39 % | **100 %** (row 255) |
| `M=320 N=1536` | 50.60 % | 50.59 % | 50.59 % | 50.61 % | 100 % | 50.39 % | **100 %** (row 319) |
| `M=192 N=8960` | 14.16 % overall | | | | | | |

Two facts worth keeping: the "last block is exact" signature is **`M = 384`-specific** — at `M = 256`
and `M = 320` all four column blocks come out equal at ~50.7 % — and the surviving interior fraction
**falls as M grows** (50.7 % at M=256/320, 25.4 % at M=384) while the **first and last row of the
tile stay 100 %**. That is a different structure from what Gate 3 recorded, and it means there is no
fixed "correct region" rule to exploit: the only shapes measured to be *entirely* correct are the
small ones above.

## Two findings

### 1. The break is the M row tile, not the multi-output-block path

`M = 1` over the **full** 1536 columns is 100 % exact, and `M = 8/128/160/192/200 × N = 384` is
100 % exact, and `M = 128 × N = 768` is 100 % exact. Multi-block output is therefore **fine**; only
row tiles of 208 rows or more go wrong. That kills the fix Gate 3 proposed ("one graph per
384-column block") as stated — **that graph is exactly the one this compiler refuses**.

### 2. The compiler has a shape hole: `OP_FAILED` in ~58 ms, in a band

The refused shapes fail in **56–73 ms** (median 58) with `OP_FAILED`, while the accepted ones take
**135–345 ms** to compile — a fast rejection, not a timeout. The hole is **not** monotonic in any
single quantity:

- refused at `N = 384` for `M ≥ 208` (but `M = 200` is fine, and `M = 384` at `N = 1536` compiles);
- refused at `N = 768` for `M ≥ 192` (but `M = 128` compiles, and `M = 384` at `N = 1536` compiles);
- refused at `N = 1536` for `M ≤ 192` (but `M = 1` compiles, and `M ≥ 256` compiles).

So both "MAC count" and "output bytes" fail to explain the set: `200×384 = 118 M MACs` compiles
while `208×384 = 123 M` does not, and `128×768 = 151 M` compiles while `192×768 = 226 M` does not,
yet `384×1536 = 906 M` compiles. **No mechanism is claimed** — the HAL gives `OP_FAILED` with no
further detail, and this phone exposes no readable TPU compiler log. What is measured is the map.

## The exact recipe (measured, not modelled)

> A bit-exact NPU stage exists on this HAL as **`N ∈ {128, 384, 768}` with `M ≤ 200`**, one graph per
> output-column block, one graph per row tile.

For a Preset-A block (384 tokens, `K = 1536`) that means, per GEMM: `ceil(384/192) = 2` row tiles ×
the GEMM's column blocks — 4 graphs for the eight 1536-wide GEMMs, and 24 (23 × 384 + 1 × 128, the
tail block at `col0 = 8832`, also measured 100 %) for `ffn.up`. Per block that is
`8 × (2 × 4) + 2 × 24 = 112` executes instead of today's 9.

**Indicative timing** (one `attn1.to_q`-shaped probe, 2 executes, no thermal gate, cold weight
fetch on the first — *not* table-grade, *not* a `T_N` claim): warm exec **1.54–1.69 ms** for
`M=192 × N=384`, 1.75–2.01 ms for `M=128 × N=768`, 1.18–1.24 ms for `M=192 × N=128`, and
**22.95 ms** for the one wide `M=192 × N=8960` graph. Extrapolating those to the 112-execute recipe
puts an exact NPU block near **0.18 s** — *below* the 377.8 ms the current 9-graph stage measures
(`RESULTS_proc_b_p33.md`). If that survives a proper measurement it means the wide graphs are slow
*and* wrong, but it is an extrapolation from 2 executes and must not enter a table.

## Consequence — reported, not applied

- **Gate 3 stays FAIL** for the configuration the pipeline runs today (9 graphs at `M=384`). A
  passing configuration now exists and its cost is unmeasured; that is the next measurement, not a
  lock change.
- `T_N` (0.4178 / 0.6484 / 0.6758 s) is unchanged as a latency figure for the *measured* graphs; the
  exact recipe needs its own `T_N`, measured the same way (repeated executes, thermal gate,
  co-run), before any cut argument may use it.
- No rate in `configs/rates_preset_a.json` edited; `k_b*` stays 160.

## What this does not prove

- **Why** the compiler refuses those shapes, and why row tiles ≥ 208 rows return garbage in the
  shapes that do compile. No mechanism is inferred from the map.
- That the exact recipe keeps its 100 % under repeated executes, other blocks (only `block_00` was
  probed), or `ffn.down` (K = 8960, a different axis, still Mali).
- Any latency claim: the exec figures above are indicative and explicitly excluded from tables.

## Thermal

Parity probes, not timing runs, so the `battery + 3` gate is not binding — but recorded anyway:
runs started at `TPU 26.0 · VIRTUAL-SKIN 23.8 · skin1 24.0 · skin2 24.4 · battery 22.9` (gate line
25.9, i.e. **0.1 °C over** at the idle floor) and the device moved < 2 °C across all five probes.
The `exec_ms` numbers are quoted as indicative for exactly this reason.
