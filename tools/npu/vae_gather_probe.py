#!/usr/bin/env python3
"""P3.19 — is the input BUILD structurally right, or was its cosine just upstream drift?

P3.18 priced every glue hop and the conv's output grid and found them worth ~0.01 dB:
the walk's 47 % rel-RMSE is not rounding. That leaves the one op class that injects
error conditioned on the *state*: the conv's input build (temporal shuffle, 2x upsample,
requant to `SA_in`). P3.17 measured `in_cos` = the walk's built input vs the reference's
own conv input — but that number conflates the wiring with everything the walk had
already accumulated.

This tool removes the drift by construction. For every conv whose state comes straight
out of another conv (no NORMSILU / ADD / LOAD in between — i.e. exactly the upsample and
shortcut chain), it rebuilds that conv's input from the PREVIOUS conv's **reference**
codes using the program's own wiring, and scores it against the reference's own conv
input. A cosine of ~0.999 means the wiring is right and P3.17's `in_cos` was drift; a
systematic ~0.95 means the wiring itself loses that much, at a clean state.

  .venv-ref/Scripts/python.exe tools/npu/vae_gather_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vae_sim as V  # the wire parser, the Sim, the reference trace — one implementation

OUT = V.OUT


def adjacencies(ops: list[tuple]) -> list[tuple[int, int]]:
    """(source conv, target conv) pairs where the state passes conv -> conv unchanged."""
    pairs, prev = [], None
    for op in ops:
        k = op[0]
        if k == "CONV":
            if prev is not None:
                pairs.append((prev, op[1]))
            prev = op[1]
        elif k in ("SAVE", "SKIP"):
            continue                       # bookkeeping / gather flags: state unchanged
        else:
            prev = None                    # NORMSILU/NORM/SILU/ADD/LOAD rewrite the state
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wire", default=str(V.WIRE))
    ap.add_argument("--latent", default=None)
    ap.add_argument("--out", default="vae_gather")
    args = ap.parse_args()

    layers, ops = V.parse_wire(Path(args.wire))
    pairs = adjacencies(ops)
    V.log(f"wire: {len(layers)} layers, {len(ops)} ops, {len(pairs)} conv->conv adjacencies: "
          + ", ".join(f"{a}->{b}" for a, b in pairs))

    ddir = Path(args.latent) if args.latent else None
    if ddir is None:
        cands = [c for c in sorted(V.ART.glob("dump_*")) if (c / "vae_latent.txt").exists()]
        if not cands:
            raise SystemExit("no dump with vae_latent.txt")
        ddir = cands[-1]
    lat, lmeta = V.latent_codes(ddir)
    V.log(f"latent {ddir.name}: {lat.shape} codes at sa {lmeta['sa']:.9g}")

    tables = V.Sim(layers, ops, 10 ** 6)          # tables only
    sa_out = {L["name"]: float(tables.SAOUT[L["ai"]]) for L in layers}
    sa_in_d = {L["name"]: float(tables.SAIN[L["ai"]]) for L in layers}
    z = (torch.from_numpy(lat.astype(np.float32) * np.float32(lmeta["sa"]))
         .permute(1, 0, 2, 3).unsqueeze(0).contiguous())
    tmp = Path(tempfile.mkdtemp(prefix="vae_gather_"))
    t0 = time.time()
    tr, _, _ = V.reference_trace(z, list(sa_out), sa_out, sa_in_d, tmp)
    V.log(f"fp32 reference in {time.time() - t0:.1f} s")

    rows = []
    ts = time.strftime("%Y%m%d_%H%M%S")
    for src, dst in pairs:
        Li, Lj = layers[dst], layers[src]
        ref = tr.get(Li["name"])
        if not ref or not ref.get("in_path"):
            V.log(f"  {src}->{dst}: no reference input stored — skipped")
            continue
        sim = V.Sim(layers, ops, 10 ** 6, shuffle="shift")
        codes = np.fromfile(tr[Lj["name"]]["path"], dtype=np.int8).reshape(tr[Lj["name"]]["shape"])
        sim.act, sim.sa = codes, float(sim.SAOUT[Lj["ai"]])
        try:
            inb = sim.gather(Li, float(sim.SAIN[Li["ai"]]))
        except SystemExit as e:                      # shape rule refused the build
            V.log(f"  {src}->{dst}: {e}")
            rows.append(dict(src=src, dst=dst, name=Li["name"], note=str(e)))
            continue
        ric = np.fromfile(ref["in_path"], dtype=np.int8).reshape(ref["in_shape"])
        if tuple(ric.shape) != tuple(inb.shape):
            V.log(f"  {src}->{dst}: shape {inb.shape} vs reference {ric.shape} — skipped")
            rows.append(dict(src=src, dst=dst, name=Li["name"],
                             note=f"shape {list(inb.shape)} vs {list(ric.shape)}"))
            continue
        c = V.cosine_codes(inb, ric)
        frac = float(np.mean(inb != ric))
        d = inb.astype(np.int16) - ric.astype(np.int16)
        per_frame = [V.cosine_codes(inb[t], ric[t]) for t in range(inb.shape[0])]
        # where the difference lives, so the cause is localisable without another reference
        # run: per frame (is it the SHUFFLE's new frames or every frame?) and its size.
        rows.append(dict(src=src, dst=dst, name=Li["name"], wiring=Li["wiring"],
                         frames=[Lj["frames"], Li["frames"]], cosine=c,
                         codes_differ=frac, sa_src=float(sim.SAOUT[Lj["ai"]]),
                         sa_in=float(sim.SAIN[Li["ai"]]),
                         mean_abs_diff=float(np.abs(d).mean()),
                         max_abs_diff=int(np.abs(d).max()),
                         within1=float(np.mean(np.abs(d) <= 1)),
                         per_frame_cos=per_frame,
                         diff_by_frame=[float(np.abs(d[t]).mean()) for t in range(d.shape[0])]))
        ddir_out = OUT / f"probe_{ts}"
        ddir_out.mkdir(exist_ok=True)
        np.savez_compressed(ddir_out / f"{src}_{dst}.npz", built=inb, ref=ric)
        V.log(f"  {src}->{dst} {Li['name'][:44]:44s} wiring {Li['wiring']} "
              f"frames {Lj['frames']}->{Li['frames']}  cosine {c:.6f}  "
              f"{frac * 100:.2f} % codes differ  |d| {np.abs(d).mean():.2f} codes "
              f"(max {np.abs(d).max()}, {np.mean(np.abs(d) <= 1) * 100:.1f} % within 1)")
        V.log("     per frame " + " ".join(f"{v:.4f}" for v in per_frame))

    doc = dict(kind="vae_gather_probe", timestamp=ts, latent_dir=ddir.name,
               pairs=len(pairs), rows=rows,
               note="Each conv whose state comes straight from another conv has its input "
                    "REBUILT from the previous conv's reference codes with the program's own "
                    "wiring (shift gather), then scored against the reference's own conv input. "
                    "Upstream drift is removed by construction, so 1 - cosine is the wiring's "
                    "own cost at a clean state.")
    js = OUT / f"{args.out}_{ts}.json"
    js.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    md = [f"# Input-build probe — {ts}", "",
          f"latent `{ddir.name}` · {len(pairs)} conv→conv adjacencies · wire "
          f"`{Path(args.wire).name}`", "",
          "| src → dst | module | wiring | frames | cosine | codes differ |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda v: "—" if v is None else f"{v:.6f}"
        md.append(f"| {r['src']} → {r['dst']} | `{r['name'].split('decoder.')[-1]}` | "
                  + (f"{r.get('wiring')}" if r.get("wiring") is not None else "—") + " | "
                  + (f"{r['frames'][0]}→{r['frames'][1]}" if r.get("frames") else "—") + " | "
                  + f(r.get("cosine")) + " | "
                  + (f"{r['codes_differ'] * 100:.2f} %" if r.get("codes_differ") is not None else
                     (r.get("note") or "—")) + " |")
    mdp = OUT / f"{args.out}_{ts}.md"
    mdp.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {js}\nwrote {mdp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
