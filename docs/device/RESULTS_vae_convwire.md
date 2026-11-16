# P3.12 — the wired decode executes: 592 `/ 592`, zero saturation, and the glue costs more than the convs

*2026-11-16 07:02. Device run `vae_wire_20261116_070210` (serial `25311JEGR11811`,
same boot as the P3.7+ session — `/proc/uptime` 16,933 s at 07:10). Tool
`tools/npu/vae_wire.py` (new) + `npu_block convwire` (new mode), plan
`docs/device/vae_wire.txt` (101 program ops, derived and asserted host-side).
**Mechanism-grade, not table-grade: the run started 0.4 °C over the
`battery+3` gate (TPU 24.0 vs 23.6, cold battery 20.6) and ended at TPU 39.0.
No device RATE is quoted from here — the numbers below are counts, walls and
saturation, and `T_V`/`T_N`/`T_G`/`k_b*` are untouched.*

## What was unrun, and why it mattered

P3.11 measured the decode's wiring and left it unexecuted: `convseq` ran the 592
`CONV_2D` dispatches with **no norm, no SiLU, no residual add**, its activation
scale table indexed by POSITION (the artifact's list order is a storage order —
`forward` applies `conv_shortcut` *first*, so 3 positions were wrong), and its
output quantised at the *consumer's* scale. Saturation was the symptom: median
0.63 %, worst **38.57 %** on `up_blocks.3.resnets.2.conv2`, and the scale ratio
explained only r = 0.50 of it — the missing ops, not the scales, were clipping
the walk.

## The program

`tools/npu/vae_wire.py` walks `vae_wiring.json` and emits a linear program in
EXECUTION order; the device walks it op by op:

| op | count | what the device does |
|---|---|---|
| `CONV` | 36 | the artifact layer's taps/frames, one graph per tap, input operand at the layer's measured `SA_in`, **output operand at its measured `SA_out`** |
| `NORMSILU` | 29 | one fused streaming pass: `x/‖x‖₂·√C·γ` per pixel over channels, then SiLU, quantised at the consumer conv's `SA_in` |
| `ADD` | 14 | residual at `post_add_sa` |
| `SAVE`/`LOAD` | 14/1 | the resnet entry / shortcut output |
| *folded* | 2+3 | temporal shuffle and 2× nearest upsample are the consumer conv's input gather (wiring codes 1/2/5) |
| `SKIP attn` | 1 | mid-block attention — **not executed**, no INT8 artifact exists (Mali price 0.0855 s) |

Op counts are asserted against the plan before the device is touched (36/29/29/
14/14/2/3/1 — all exact), the artifact index is asserted to be a **permutation**
(it is what the scale tables are indexed by), and every conv's wiring is derived
from the activation's shape, so a mismatch is a hard error rather than a wrong
execute. Layout is int8 throughout at a declared scale — the largest activation
is 9×96×272×480 = 112.8 MB int8, so a float activation copy was never an option.

## Result

**36/36 convs, 592/592 executes, `pc=1` accepted on all 36 (0 fallbacks, 0 tap
failures, 0 fatals), 100 taps, 101 program ops, 29 NORMSILU + 14 ADD passes.**

| quantity | value |
|---|---|
| `CONV_2D` executes | **592 / 592** (= the plan's count) |
| sum of execute walls (TS) | **9,056.3 ms** (15.30 ms mean, 44.56 ms max) |
| P3.10's same sum | 8,822.6 ms ⇒ **+2.6 %**, consistent with this walk heating 24 → 39 °C (the decode's measured temperature term is +1.7 % median) |
| conv layer walls (incl. per-tap input copy + epilogue) | 12,189.7 ms |
| **norm+SiLU passes** | **10,043.5 ms for 1.238 G elements** (123 M elem/s) |
| residual adds | 1,070.1 ms for 0.566 G elements |
| input build + quantise | 630.7 ms |
| output dequant/clamp | 1,195.6 ms |
| compile | 18.79 s (100 graphs) |
| whole walk | 44.86 s |
| **saturation** | **0.0841 % overall; 19/36 layers clamp nothing; worst layer 0.0721 %** |

### The two findings

1. **The wiring removes the clipping.** P3.10's worst layer saturated 38.57 % of
   its output at `1/127`-era scales; with the norm in place and the add at
   `post_add_sa`, the same decode clamps **0.0721 % at worst and nothing on
   19/36 layers**. That is the range argument P3.11 predicted — a residual add at
   conv2's scale *had* to clip (post-add range is 1.03×…4.76× conv2's own), and
   the 89× step between `conv1` and `conv2` is the norm, not a rescale. It is a
   **range** result, not a quality claim: no output was compared to the fp32
   reference, so "it decodes" is still unproven (next action).
2. **The glue costs more than the convs.** 9.06 s of `CONV_2D` against
   10.04 s of norm+SiLU + 1.07 s of adds + 1.83 s of prep/epilogue. The wiring
   is a **scalar C++ reference implementation**, not a priced kernel: P3.11's
   note that the norm cannot be folded into a conv epilogue (its scale is
   per-pixel) is what forces a separate pass, so `T_V` = 8.92 s still holds only
   if that pass is fused or replaced — exactly the caveat P3.8c attached to the
   harness epilogue, now with a second term.

### Correctness of the wiring, as opposed to its execution

The host derivation proves the *shape* chain (conv → norm → conv → add →
shuffle → conv closes for all 36 layers, asserted, not asserted-away), and the
two cross-checks P3.11 found reproduce here: `layer 12`'s `prep_ms` = 31.9 and
`layer 21`'s = 169.0 are the two wiring-2 gathers (3×768×34×60 → 5×384×68×120 and
5×768×68×120 → 9×384×136×240), i.e. the fold is doing the work it claims.
`sa_out[i] == sa_in[i+1]` is not asserted here — the program carries both
independently, which is why the add needed its own `post_add_sa`.

## Defect found and fixed en route

`vae_wiring.json`'s `gamma_off` is in **BYTES**, not floats (the same trap the
weight-scale offsets paid for in P3.8a). Read as floats it put the last
`norm_out` gamma 4× past the end of a 7,776-float blob, and the device's own
range check caught it as `FATAL|wire_norm_gamma_range|goff=7680|C=384|ngam=7776`
on the first full attempt. It is now divided exactly once, with an f4-alignment
assertion and a length check against `n_gamma_floats`.

## What is still not done

- **No quality number.** The walk executes in the right numeric range; whether
  it *decodes* needs the device's output compared against the fp32 decode from
  the same quantised latent. The program has no dump op yet — that is the next
  bounded unit (`OP DUMP` writing the final 9×3×272×480 codes, then a host PSNR
  against `.venv-ref`).
- **Mechanism-grade thermal.** A table-grade re-run needs `tools/npu/cool_rerun.py`
  (15–20 min to the floor) and cannot make `T_V` quotable until the glue is not
  scalar C++.
- The mid-block attention is still skipped (Mali, 0.0855 s/chunk) — the walk is
  faithful everywhere else, and this is a stated deviation.
