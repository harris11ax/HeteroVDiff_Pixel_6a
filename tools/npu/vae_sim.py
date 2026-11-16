#!/usr/bin/env python3
"""P3.14 — the host-side program simulator: which GLUE op in `convwire` is wrong?

P3.13 exonerated the conv: with the tap→frame labels mirrored (fixed), layer 0's
device codes are 98.59 % bit-identical to an exact emulation of the program, so
the remaining decode error (11.60 dB / cosine 0.219 against fp32) is in the ops
*between* the convs — NORMSILU, the residual ADD, SAVE/LOAD, or the temporal
shuffle / 2× upsample gathers folded into each conv's input build.

This tool walks `docs/device/vae_wire.txt` op-for-op with the device's own
semantics (int8 codes, per-tensor activation scales, per-tap int8 rounding, the
same gather arithmetic as `npu_block.cpp`) from the same latent the device
consumed, and diffs **each conv's output** against the fp32 reference's own
intermediate tensor for the same module. The first layer where the cosine
collapses localizes the defect: every op between conv L−1 and conv L becomes the
candidate set, and nothing after L matters yet.

Why a simulator and not another device probe: a device probe shows one layer's
input and output per run, and the conv is already proven faithful. The
simulator costs no phone time, so a hypothesis (e.g. the shuffle mapping) can be
re-run in seconds instead of a 45 s walk.

  .venv-ref/Scripts/python.exe tools/npu/vae_sim.py --max-layer 12
  .venv-ref/Scripts/python.exe tools/npu/vae_sim.py                  # all 36
  .venv-ref/Scripts/python.exe tools/npu/vae_sim.py --device artifacts/vae/dump_<ts>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts" / "vae"
VAE_DIR = REPO / "weights_raw" / "DCM_WAN" / "vae"
WIRE = REPO / "docs" / "device" / "vae_wire.txt"
OUT = REPO / "docs" / "device"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------- the wire file
def parse_wire(p: Path) -> tuple[list[dict], list[tuple]]:
    layers: list[dict] = []
    ops: list[tuple] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line or line[0] == "#":
            continue
        f = line.split()
        if f[0] == "LAYER":
            layers.append(dict(idx=int(f[1]), frames=int(f[2]), fin=int(f[3]),
                               cout=int(f[4]), cin=int(f[5]), h=int(f[6]), w=int(f[7]),
                               ntaps=int(f[8]), wiring=int(f[9]), sc_off=int(f[10]),
                               ai=int(f[11]), name=f[12], taps=[]))
        elif f[0] == "TAP":
            layers[-1]["taps"].append(dict(tap=int(f[1]), delta=int(f[2]), off=int(f[3]),
                                           nbytes=int(f[4]), kh=int(f[5]), kw=int(f[6])))
        elif f[0] == "OP":
            k = f[1]
            if k in ("CONV", "SAVE", "LOAD"):
                ops.append((k, int(f[2])))
            elif k in ("NORMSILU", "NORM"):
                ops.append((k, int(f[2]), int(f[3]), float(f[4]), float(f[5])))
            elif k == "SILU":
                ops.append((k, float(f[2])))
            elif k == "ADD":
                ops.append((k, int(f[2]), float(f[3]), float(f[4])))
            else:
                ops.append((k, f[2]))
    return layers, ops


def rnd_half_away(x: np.ndarray) -> np.ndarray:
    """NAPI rounding: half away from zero (`np.rint` is half-to-even)."""
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def q_clamp(v: np.ndarray) -> tuple[np.ndarray, int]:
    q = rnd_half_away(np.asarray(v, dtype=np.float64))
    n = int(np.count_nonzero((q > 127) | (q < -127)))
    return np.clip(q, -127, 127).astype(np.int8), n


def read_meta(p: Path) -> dict:
    d: dict = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition("=")
        if k:
            d[k] = v
    for k in ("f", "c", "h", "w"):
        if k in d:
            d[k] = int(float(d[k]))
    if "sa" in d:
        d["sa"] = float(d["sa"])
    return d


# ----------------------------------------------------------------- the simulator
class Sim:
    """`npu_block convwire`'s walk, op for op, in numpy."""

    def __init__(self, layers: list[dict], ops: list[tuple], max_layer: int,
                 shuffle: str = "wire"):
        self.layers, self.ops, self.max_layer = layers, ops, max_layer
        self.shuffle = shuffle
        self.wbytes = np.memmap(ART / "weights_int8.bin", dtype=np.int8, mode="r")
        self.SC = np.fromfile(ART / "scales_f32.bin", dtype="<f4").astype(np.float64)
        self.BI = np.fromfile(ART / "bias_f32.bin", dtype="<f4").astype(np.float64)
        sa2 = np.fromfile(ART / "act_scales2_f32.bin", dtype="<f4")
        self.nlay = sa2.size // 2
        self.SAIN = sa2[:self.nlay].astype(np.float64)
        self.SAOUT = sa2[self.nlay:].astype(np.float64)
        self.NM = np.fromfile(ART / "norms_f32.bin", dtype="<f4")
        assert self.nlay == len(layers), f"sa2 {self.nlay} vs {len(layers)} layers"
        self.act: np.ndarray | None = None       # int8 [f, c, h, w]
        self.sa = 0.0
        self.slot: list[tuple | None] = [None, None]
        self.hist: list[tuple] = []              # the last two conv outputs
        self.clamps = 0
        self.rec: list[dict] = []

    def weights(self, li: int) -> list[np.ndarray]:
        """OHWI [Cout, kh, kw, Cin] -> conv2d's [Cout, Cin, kh, kw]."""
        L = self.layers[li]
        out = []
        for T in L["taps"]:
            co, ci, kh, kw = L["cout"], L["cin"], T["kh"], T["kw"]
            assert T["nbytes"] == co * ci * kh * kw, \
                f"{L['name']} tap {T['tap']}: {T['nbytes']} B != {co}x{ci}x{kh}x{kw}"
            raw = np.asarray(self.wbytes[T["off"]:T["off"] + T["nbytes"]]).reshape(co, kh, kw, ci)
            out.append(np.ascontiguousarray(raw.transpose(0, 3, 1, 2)))
        return out

    def gather_shift(self, L: dict, sa_in: float) -> np.ndarray:
        """The temporal-upsample gather MEASURED off the reference (P3.14).

        `WanResample.forward` emits, for a T-frame buffer b (the time_conv output):

            frame 0        = b[0], channel half 0 only
            frames 2j-1,2j = b[j] halves 0 and 1        (j = 1 .. T-1)

        i.e. 1 + 2(T-1) = 2T-1 frames. `npu_block.cpp` implements `(t//2, t%2)`
        instead, which emits both halves of b[0] and then drops the last half of
        b[T-1]: same frame COUNT, different content — so it fails silently. The
        mapping was measured, not read off the model: correlating the reference's
        own `resample.1` Conv2d input against every candidate frame gives
        [b0h0, b1h0, b1h1, b2h0, b2h1] at 0.896 / 0.977 / 0.972 / 0.970 / 0.963,
        while the wire's own choice for frame 1 (b0h1) scores 0.255.
        """
        act = self.act
        fin, cin, h, w = L["fin"], L["cin"], L["h"], L["w"]
        spatial = L["wiring"] in (2, 5)
        g = float(self.sa) / float(sa_in)
        ih = np.arange(h) // 2 if spatial else np.arange(h)
        iw = np.arange(w) // 2 if spatial else np.arange(w)
        inb = np.zeros((fin, cin, h, w), dtype=np.int8)
        for t in range(fin):
            if t == 0:
                srcf, half = 0, 0
            else:
                srcf, half = (t + 1) // 2, (t + 1) % 2
            if srcf >= act.shape[0]:
                continue
            sub = act[srcf][half * cin:(half + 1) * cin][:, ih][:, :, iw]
            if g == 1.0 and L["wiring"] in (0, 1, 2, 5):
                inb[t] = sub
            else:
                inb[t], _ = q_clamp(sub.astype(np.float64) * g)
        return inb

    def gather(self, L: dict, sa_in: float) -> np.ndarray:
        """The conv's input codes at `sa_in` — the device's wiring step, vectorised."""
        if self.shuffle == "shift" and L["wiring"] in (1, 2):
            return self.gather_shift(L, sa_in)
        act = self.act
        assert act is not None, f"{L['name']}: no activation"
        fin, cin, h, w = L["fin"], L["cin"], L["h"], L["w"]
        tshuf = L["wiring"] in (1, 2)
        spatial = L["wiring"] in (2, 5)
        g = float(self.sa) / float(sa_in)
        ih = np.arange(h) // 2 if spatial else np.arange(h)
        iw = np.arange(w) // 2 if spatial else np.arange(w)
        want_f = (2 * act.shape[0] - 1) if tshuf else act.shape[0]
        want_c = (2 * cin) if tshuf else cin
        want_h = (2 * act.shape[2]) if spatial else act.shape[2]
        want_w = (2 * act.shape[3]) if spatial else act.shape[3]
        if (want_f, want_c, want_h, want_w) != (fin, act.shape[1], h, w):
            raise SystemExit(
                f"{L['name']}: wiring {L['wiring']} cannot build {fin}x{cin}x{h}x{w} from "
                f"act {act.shape} (want f={want_f} c={want_c} h={want_h} w={want_w})")
        inb = np.zeros((fin, cin, h, w), dtype=np.int8)
        for t in range(fin):
            srcf = t // 2 if tshuf else t
            if srcf >= act.shape[0]:
                continue                              # device: memset 0
            half = (t % 2) * cin if tshuf else 0
            sub = act[srcf][half:half + cin][:, ih][:, :, iw]
            if L["wiring"] == 0 and g == 1.0:
                inb[t] = sub
            else:
                inb[t], _ = q_clamp(sub.astype(np.float64) * g)
        return inb

    def conv(self, li: int) -> dict:
        L = self.layers[li]
        ai = L["ai"]
        sa_in, sa_out = float(self.SAIN[ai]), float(self.SAOUT[ai])
        sc = self.SC[L["sc_off"]:L["sc_off"] + L["cout"]]
        bi = self.BI[L["sc_off"]:L["sc_off"] + L["cout"]]
        # the C++ builds the INT32 bias in ACCUMULATOR units, then lets CONV_2D
        # requant: out = (acc + bias_int) * sa_in*sc[c] / sa_out
        bi_int = np.trunc(bi / (sa_in * sc)).astype(np.int64)
        req = (sa_in * sc / sa_out).reshape(-1, 1, 1).astype(np.float64)
        W = self.weights(li)
        inb = self.gather(L, sa_in)
        pad = (L["taps"][0]["kh"] // 2, L["taps"][0]["kw"] // 2)
        # exact integer accumulation: keep a partial sum below 2^24 in float32
        per_term = 127 * 127 * (2 * pad[0] + 1) * (2 * pad[1] + 1)
        step = max(1, int((2 ** 24) // max(1, per_term)))
        outb = np.zeros((L["frames"], L["cout"], L["h"], L["w"]), dtype=np.int8)
        clamps = 0
        for f in range(L["frames"]):
            acc = np.zeros((L["cout"], L["h"], L["w"]), dtype=np.int32)
            for ti, T in enumerate(L["taps"]):
                src = f - T["delta"]
                if src < 0 or src >= L["fin"]:
                    continue
                x = torch.from_numpy(inb[src].astype(np.float32))
                acc_t = np.zeros((L["cout"], L["h"], L["w"]), dtype=np.int32)
                for c0 in range(0, L["cin"], step):
                    c1 = min(L["cin"], c0 + step)
                    wch = torch.from_numpy(W[ti][:, c0:c1].astype(np.float32))
                    part = F.conv2d(x[None, c0:c1], wch, padding=pad)[0]
                    acc_t += torch.round(part).to(torch.int32).numpy()
                # P3.13 MEASURED DEFECT (not fixed here): every tap's graph carries
                # the same INT32 bias and each tap is rounded to int8 before the sum.
                v = (acc_t.astype(np.float64) + bi_int.reshape(-1, 1, 1)) * req
                q, n = q_clamp(v)
                clamps += n
                acc += q.reshape(L["cout"], L["h"], L["w"]).astype(np.int32)
            q2, n2 = q_clamp(acc)
            clamps += n2
            outb[f] = q2.reshape(L["cout"], L["h"], L["w"])
        self.clamps += clamps
        self.act = outb
        self.sa = sa_out
        self.hist.append((outb, sa_out, li))
        self.hist = self.hist[-3:]
        self.last_inb = inb
        self.last_inb_name = L["name"]
        r = dict(idx=li, ai=ai, name=L["name"], shape=list(outb.shape), sa_out=sa_out,
                 clamp=clamps / outb.size)
        self.rec.append(r)
        return r

    # ---------------------------------------------------------------- the float ops
    def normsilu(self, goff: int, C: int, gam_scale: float, out_sa: float,
                 do_norm: bool, do_silu: bool) -> None:
        a = self.act
        assert a.shape[1] == C, f"norm C={C} vs act {a.shape[1]}"
        sain = np.float32(self.sa)
        x = a.astype(np.float32) * sain
        if do_norm:
            den = np.sqrt(np.sum(x * x, axis=1, keepdims=True, dtype=np.float32))
            den = np.maximum(den, np.float32(1e-12))      # F.normalize's clamp_min
            y = x / den * np.float32(gam_scale) * self.NM[goff:goff + C][None, :, None, None]
        else:
            y = x
        if do_silu:
            y = y / (np.float32(1.0) + np.exp(-y))
        q, n = q_clamp(y / np.float32(out_sa))
        self.clamps += n
        self.act = q.reshape(a.shape)
        self.sa = float(out_sa)

    def add(self, slot: int, out_sa: float) -> None:
        s, s_sa = self.slot[slot & 1]
        assert s.shape == self.act.shape, f"ADD act {self.act.shape} vs slot {s.shape}"
        v = (self.act.astype(np.float64) * self.sa + s.astype(np.float64) * s_sa) / out_sa
        q, n = q_clamp(v)
        self.clamps += n
        self.act = q.reshape(self.act.shape)
        self.sa = float(out_sa)

    # ---------------------------------------------------------------- the walk
    def run(self, lat: np.ndarray, lat_sa: float) -> None:
        self.act, self.sa = lat, float(lat_sa)
        for op in self.ops:
            k = op[0]
            if k == "CONV":
                li = op[1]
                if li > self.max_layer:
                    break
                r = self.conv(li)
                log(f"[{li:2d}] ai={r['ai']:2d} {r['name'][:46]:46s} {r['shape']} "
                    f"sa={r['sa_out']:.4g} clamp={r['clamp'] * 100:.4f}%")
            elif k == "SAVE":
                self.slot[op[1] & 1] = (self.act, self.sa)
            elif k == "LOAD":
                s = self.slot[op[1] & 1]
                if s is None:
                    raise SystemExit(f"LOAD of empty slot {op[1]}")
                self.act, self.sa = s
            elif k in ("NORMSILU", "NORM"):
                self.normsilu(op[1], op[2], op[3], op[4], True, k == "NORMSILU")
            elif k == "SILU":
                self.normsilu(0, self.act.shape[1], 1.0, op[1], False, True)
            elif k == "ADD":
                self.add(op[1], op[2])


def latent_codes(ddir: Path) -> tuple[np.ndarray, dict]:
    meta = read_meta(ddir / "vae_latent.txt")
    codes = np.fromfile(ddir / "vae_latent.bin", dtype=np.int8)
    want = meta["f"] * meta["c"] * meta["h"] * meta["w"]
    if codes.size != want:
        raise SystemExit(f"{ddir.name}: {codes.size} latent codes vs meta {want}")
    a = codes.reshape(meta["f"], meta["h"], meta["w"], meta["c"]).transpose(0, 3, 1, 2)
    return np.ascontiguousarray(a), meta


# ----------------------------------------------------------------- the reference
def reference_trace(z: torch.Tensor, names: list[str], sa_out: dict[str, float],
                    sa_in: dict[str, float], tmp: Path) -> tuple[dict, torch.Tensor]:
    """fp32 decode with the program's deviations (attention skipped, post_quant_conv
    bypassed), each layer module's output concatenated over the decoder's frame
    chunks and written as int8 codes at that layer's own `SA_out`."""
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(str(VAE_DIR), torch_dtype=torch.float32).eval()
    calls: dict[str, list] = {n: [] for n in names}
    ins: dict[str, list] = {n: [] for n in names if n.endswith(".resample.1")}
    hooks = []
    for n in names:
        mod = vae.get_submodule(n)
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, n=n: calls[n].append(o.detach().clone())))
        if n in ins:                       # the Conv2d's own input = the gathered, shuffled,
            hooks.append(mod.register_forward_pre_hook(     # upsampled tensor under test
                lambda m, i, n=n: ins[n].append(i[0].detach().clone())))
    hooks.append(vae.decoder.mid_block.attentions[0].register_forward_hook(lambda m, i, o: i[0]))
    hooks.append(vae.post_quant_conv.register_forward_hook(lambda m, i, o: i[0]))
    with torch.no_grad():
        out = vae.decode(z).sample
    for h in hooks:
        h.remove()
    tr = {}
    for n in names:
        parts = calls[n]
        if not parts:
            tr[n] = None
            continue
        if parts[0].dim() == 4:                        # a Conv2d: [b*t, c, h, w]
            cat = torch.cat(parts, dim=0)              # frames are already leading
        else:                                          # [b, c, t, h, w]
            cat = torch.cat(parts, dim=2).squeeze(0).permute(1, 0, 2, 3)
        cat = cat.contiguous()
        sa = float(sa_out[n])
        q = torch.clamp(torch.round(cat / sa), -127.0, 127.0).to(torch.int8)
        p = tmp / f"{n}.i8"
        p.write_bytes(q.numpy().tobytes())
        tr[n] = dict(path=str(p), shape=list(q.shape), sa=sa,
                     clamp=float((cat / sa).abs().gt(127).float().mean()))
        if n in ins:
            ic = torch.cat(ins[n], dim=0)              # [frames, C, H, W] in output-frame order
            p = tmp / f"{n}.in.i8"
            p.write_bytes(torch.clamp(torch.round(ic / float(sa_in[n])), -127.0, 127.0)
                          .to(torch.int8).numpy().tobytes())
            tr[n]["in_path"] = str(p)
            tr[n]["in_sa"] = float(sa_in[n])
            tr[n]["in_shape"] = list(ic.shape)
    return tr, out


def cosine_codes(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wire", default=str(WIRE))
    ap.add_argument("--max-layer", type=int, default=10 ** 6)
    ap.add_argument("--latent", default=None, help="a dump_<ts> dir (default: newest)")
    ap.add_argument("--device", default=None,
                    help="a dump_<ts> dir whose vae_dump.bin is the device's final decode")
    ap.add_argument("--shuffle", default="both", choices=["wire", "shift", "both"],
                    help="temporal-upsample gather: `wire` = npu_block.cpp's (t//2, t%2); "
                         "`shift` = the mapping measured off the reference's own conv input; "
                         "`both` measures them side by side")
    ap.add_argument("--out", default="vae_sim")
    args = ap.parse_args()

    layers, ops = parse_wire(Path(args.wire))
    log(f"wire: {len(layers)} layers, {len(ops)} ops")

    if args.latent:
        ddir = Path(args.latent)
    else:                                   # newest dump that carries a full walk's latent
        cands = sorted(ART.glob("dump_*"))
        cands = [c for c in cands if (c / "vae_latent.txt").exists()]
        if not cands:
            raise SystemExit("no dump with vae_latent.txt — run `vae_wire.py --run --dump`")
        ddir = cands[-1]
    lat, lmeta = latent_codes(ddir)
    log(f"latent {ddir.name}: {lat.shape} codes at sa {lmeta['sa']:.9g}")

    sim = Sim(layers, ops, args.max_layer)          # tables only (SA_in/SA_out)

    tmp = Path(tempfile.mkdtemp(prefix="vae_sim_"))
    z = (torch.from_numpy(lat.astype(np.float32) * np.float32(lmeta["sa"]))
         .permute(1, 0, 2, 3).unsqueeze(0).contiguous())
    sa_out = {L["name"]: float(sim.SAOUT[L["ai"]]) for L in layers
              if L["idx"] <= args.max_layer}
    names = list(sa_out)
    t0 = time.time()
    sa_in_d = {L["name"]: float(sim.SAIN[L["ai"]]) for L in layers if L["idx"] <= args.max_layer}
    tr, out = reference_trace(z, names, sa_out, sa_in_d, tmp)
    log(f"fp32 reference in {time.time() - t0:.1f} s -> {tuple(out.shape)}")

    modes = ["wire", "shift"] if args.shuffle == "both" else [args.shuffle]
    results: dict[str, dict] = {}
    for mode in modes:
        log(f"--- shuffle={mode}")
        s = Sim(layers, ops, args.max_layer, shuffle=mode)
        rows, first_bad = [], None
        orig_conv = s.conv

        def conv_and_cmp(li: int, s=s, rows=rows) -> dict:
            nonlocal first_bad
            r = orig_conv(li)
            ref = tr.get(r["name"])
            if ref is None:
                r.update(cosine=None, note="no reference tensor")
            else:
                ref_codes = np.fromfile(ref["path"], dtype=np.int8).reshape(ref["shape"])
                if tuple(ref["shape"]) != tuple(r["shape"]):
                    r.update(cosine=None, ref_shape=ref["shape"],
                             note="SHAPE MISMATCH — the program's tensor is not this module's")
                else:
                    c = cosine_codes(s.act, ref_codes)
                    r.update(cosine=c, ref_clamp=ref["clamp"], note="")
                    if ref.get("in_path") and s.last_inb_name == r["name"]:
                        ric = np.fromfile(ref["in_path"], dtype=np.int8).reshape(ref["in_shape"])
                        if tuple(ric.shape) == tuple(s.last_inb.shape):
                            ci = cosine_codes(s.last_inb, ric)
                            r["cos_gather"] = ci
                            log(f"    gather vs the reference's own conv input: cosine {ci:.6f}")
                    if first_bad is None and c < 0.99:
                        first_bad = dict(idx=li, name=r["name"], cosine=c)
                    log(f"  [{li:2d}] {r['name'][:40]:40s} cosine {c:.6f}  "
                        f"clamp {r['clamp'] * 100:.4f}%  (ref clamp {ref['clamp'] * 100:.3f}%)")
            rows.append(r)
            return r

        s.conv = conv_and_cmp                        # type: ignore[method-assign]
        s.run(lat, lmeta["sa"])
        results[mode] = dict(convs=rows, first_divergence=first_bad,
                             total_clamps=int(s.clamps))
    shutil.rmtree(tmp, ignore_errors=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    doc = dict(kind="vae_program_sim", timestamp=ts, wire=str(Path(args.wire).name),
               latent_dir=ddir.name, max_layer=args.max_layer, results=results,
               note="cosine = the simulator's int8 codes vs the fp32 reference's own module "
                    "output quantised at the same SA_out; a collapse localizes the op between "
                    "the previous conv and this one")
    js = OUT / f"{args.out}_{ts}.json"
    js.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    md = [f"# VAE program simulator — {ts}", "",
          f"wire `{Path(args.wire).name}` · latent `{ddir.name}` · max layer {args.max_layer}",
          "", "`wire` = the gather `npu_block.cpp` implements; `shift` = the mapping measured "
          "off the reference decoder's own conv input. Same latent, artifact and scales.", "",
          "| layer | ai | module | shape | clamp % (wire) | cos (wire) | cos (ref) |",
          "|---|---|---|---|---|---|---|"]
    base = results[modes[0]]["convs"]
    for i, r in enumerate(base):
        other = results["shift"]["convs"][i] if "shift" in results else None
        cw = r.get("cosine")
        cr = other.get("cosine") if other else None
        md.append(f"| {r['idx']} | {r['ai']} | `{r['name']}` | {r['shape']} | "
                  f"{r['clamp'] * 100:.4f} | "
                  + ("—" if cw is None else f"{cw:.6f}") + " | "
                  + ("—" if cr is None else f"{cr:.6f}") + " |")
    for mode in modes:
        fb = results[mode]["first_divergence"]
        md += ["", f"**{mode}**: first divergence " + (
            "none (all convs >= 0.99)" if not fb else
            f"layer {fb['idx']} (`{fb['name']}`) at cosine {fb['cosine']:.6f}")]
    mdp = OUT / f"{args.out}_{ts}.md"
    mdp.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {js}\nwrote {mdp}")
    for mode in modes:
        fb = results[mode]["first_divergence"]
        print(f"{mode:5s}: " + ("no divergence" if not fb else
              f"layer {fb['idx']} {fb['name']} cosine {fb['cosine']:.6f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
