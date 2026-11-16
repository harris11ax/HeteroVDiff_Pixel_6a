# P3.1 chunk-handoff selftest — 20261115_220306

serial `25311JEGR11811` · ring `/data/local/tmp/hv_handoff.bin` · one file, 2 slots, chunk 4,700,160 B (M=1530 × d=1536 fp16), slot stride 4,702,208 B

Two processes (real PIDs) share one `MAP_SHARED` mapping; the producer stamps every 64-bit word with a chunk-derived pattern, the consumer verifies every word in order.

## Verdict

- **PASS** — chunks=200, stall=0 µs: bad_words=0, out-of-order=0/0, publish→consume mean 680.9 µs (max 1023.4 µs), producer blocked on 198/200 chunks
- **PASS** — chunks=200, stall=1000 µs: bad_words=0, out-of-order=0/0, publish→consume mean 2541.4 µs (max 8167.2 µs), producer blocked on 198/200 chunks

## Ring

- device file size **9,408,512 B** vs expected **9,408,512 B** ⇒ MATCH (one buffer, two slots)
- the ring carries residual activations only — it holds no weights, so the single-INT8-copy rule is untouched by construction

## Thermal (recorded, not a gate)

- before: TPU=25.0 VIRTUAL-SKIN=22.7 battery=21.8 skin_therm1=22.9 skin_therm2=23.2
- after:  TPU=25.0 VIRTUAL-SKIN=22.7 battery=21.8 skin_therm1=22.9 skin_therm2=23.2

## What this proves / does not

- proves: the publish (RELEASE) / spin (ACQUIRE) / release-done protocol delivers every chunk intact and in order across two processes, and the 2-slot ring back-pressures the producer instead of tearing a payload
- does not prove: steady-state period or overlap (that is P3.4/P3.5, with the real stages), nor anything about the weight artifact
