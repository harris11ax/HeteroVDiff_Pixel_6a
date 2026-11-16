# Gate 3 — numerical parity of the NPU stage — 2026-11-16

Status: **FAIL.** The Edge-TPU `FULLY_CONNECTED` graphs return the exact integer
product for only **25.8 %** of each output tile. The reference model is proved
correct by the very same dumps (row 0 and the last column block come out
bit-exact), so this is a property of the HAL's compute/copy-back path, not of the
host reference.

Tools: `tools/pipe/gate3_parity.py` (host verifier) + `proc_b --dump-parity DIR
--act payload|basis|window` (device dump). Captures
`gate3_parity_20261116_005432` (payload), `…_005910` / `…_010101` (basis, two runs
including a 4096-byte-aligned output buffer — **bit-identical**).

## What the comparison is

`proc_b` builds one `FULLY_CONNECTED` per GEMM with **per-tensor** scales
`SA = SW = 1/127` (`FULLY_CONNECTED` rejects `TENSOR_QUANT8_SYMM_PER_CHANNEL` on
this HAL — `RESULTS_npu_tn.md` §2), zero bias, output scale `SO = SA`. NNAPI's
requant then reduces to

```
acc_i64 = Σ_k a[k]·w[k]                        (exact, |acc| < 2^31)
out_i8  = clamp(round(acc_i64 · SA·SW/SO)) = clamp(round(acc/127))
```

so the whole graph is checkable in the **integer domain**. The host recomputes
`acc` exactly — int8×int8 products are exact in fp32 and partial sums over
K-split blocks of 512 are ≤ 8.32e6 < 2^24, so an fp32 BLAS GEMM is an exact
integer computation, accumulated in int64. Verified against brute-force int64
`@` including the worst case (all −128: 146,800,640 exact).

## Measured (basis activations: `A[r][r] = 127`, so `out[r][n] = W[n][r]` exactly)

| GEMM | N | K | bit-exact | max abs Δ (LSB) | rel L2 |
|---|---|---|---|---|---|
| `attn1.to_q` | 1536 | 1536 | 25.828 % | 252 | 121.6 % |
| `attn1.to_k` | 1536 | 1536 | 25.904 % | 249 | 122.0 % |
| `attn1.to_v` | 1536 | 1536 | 26.026 % | 229 | 123.8 % |
| `attn1.to_out` | 1536 | 1536 | 25.923 % | 246 | 122.2 % |
| `attn2.to_q` | 1536 | 1536 | 26.039 % | 234 | 123.9 % |
| `attn2.to_k` | 1536 | 1536 | 25.889 % | 243 | 122.2 % |
| `attn2.to_v` | 1536 | 1536 | 25.881 % | 238 | 121.3 % |
| `attn2.to_out` | 1536 | 1536 | 25.975 % | 238 | 122.3 % |
| `ffn.up` | 8960 | 1536 | 4.542 % | 235 | 139.4 % |

`ffn.down` is not in this table: it runs on Mali (`--down skip`), it is never
dumped by this binary, and its parity is a separate measurement.

The requant rounding mode was scored, not assumed: half-away-from-zero,
half-even and truncate all give the same ordering, because the error is not a
rounding artifact.

## Where the correct data is — the structural finding

Correctness per 384-column output block, all 384 rows of the tile
(`attn1.to_q`, N = 1536 → 4 blocks):

| col block | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| correct | 1.1 % | 1.1 % | 1.1 % | **100.0 %** |

`ffn.up` (N = 8960 → 24 blocks): blocks 0–21 are **1.3–1.4 %**, block 22 is
**45.1 %**, block 23 is **91.9 %**.

Per row, `attn1.to_q`:

| row | blk 0 | blk 1 | blk 2 | blk 3 |
|---|---|---|---|---|
| 0 | **100** | **100** | **100** | **100** |
| 1 | 0.3 | 0.5 | 0.5 | **100** |
| 2 | 0.3 | 0.5 | 0.8 | **100** |
| 3 | 0.8 | 0.5 | 0.5 | **100** |
| 100 | 0.8 | 1.0 | 1.0 | **100** |
| 383 | 1.0 | 0.3 | 0.3 | **100** |

So: **the last column block is exact for every row, row 0 is exact for every
column, and nothing else is.** 25.8 % of the tile. `ffn.up`'s last two blocks
being 45 %/92 % rather than 100 % says the "correct region" is itself not a clean
rule at N = 8960.

## Ruled out (each a measurement, not an argument)

| hypothesis | test | result |
|---|---|---|
| my reference is wrong | row 0 / last block are bit-exact against it | reference **correct** |
| output written asynchronously (dump raced it) | re-dump 150 ms later, `PARITY2` | `changed=0`, bytes final |
| output buffer alignment | 4096-byte-aligned static output buffer | **bit-identical** results |
| output transposed `[N,M]` | compare against `W.T` layout | 25.8 %, not a layout |
| output row stride padded | stride sweep 1536…9216 | none matches |
| rows/cols permuted | best-match search over all 1536 weight rows **and** columns | own row/col wins, at 25 % |
| element shifted in `n` | shift search δ ∈ [0,384] | max 1.4 % |
| stale buffer from another GEMM | garbage segment vs all 9 GEMMs' expected values | **no match** |
| garbage is weight bytes | verbatim search of the 46 MB block | **not present** |

The garbage carries weight-like statistics (absmean 37.7 vs 38.1) but is not any
synthesised from these weights — i.e. it is not an FC result at all.

## Consequence — reported, not applied

- `proc_b.cpp` has always declared this stage **latency-only, not a parity
  claim**, so `T_N` = 0.4178 / 0.6484 / 0.6758 s and the 204–289 GOPS figures
  stand as *latency on the real weight bytes*. What is new is that the stage does
  not currently produce the block's numerical result, so it cannot be wired into
  a functional pipeline (Gate 3 / the deliverable's "working pipeline") as is.
- Open branch, not claimed: if the HAL **computes** only what it writes
  correctly, the implied GOPS would be ~4× optimistic and `T_N` ~4× low; if it
  computes all four blocks and only the **copy-back** is partial, the latencies
  are unaffected. Nothing measured so far distinguishes these two.
- No locked decision is touched: the cut stays at `k_b` = 160 and no rate in
  `configs/rates_preset_a.json` is edited.

## Next action

Bisect the HAL with a **single-output-tile** graph: one `FULLY_CONNECTED` with
`N = 384` (one block) at the same `M = TILE`, basis activations. If that is
100 % exact, the break is in the multi-output-block path and the fix is a
per-block graph (4 compilations instead of 1) — which also re-prices `T_N`
honestly. If it is still 25 %, the break is in the `M = 384` row path and the
next probe is `M = 1`.

## Thermal (parity run, not a timing table)

`TPU 28.0 °C · VIRTUAL-SKIN 25.6 · skin1 25.8 · battery 24.4` → gate line
27.4 °C, i.e. **0.6 °C over** the `battery + 3` rule. Irrelevant to a parity
comparison, which is bit-exactness and not a timeout-sensitive timing; the
latency table above is not a rate measurement and is not quoted as one.
