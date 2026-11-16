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
import math
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
                 shuffle: str = "wire", bias_mode: str = "per_tap",
                 round_mode: str = "per_tap", label: str = "wire",
                 add_fp: tuple = (), cout_pc: bool = False, oracle: bool = False,
                 glue_fp: tuple = (), conv_out_fp: bool = False):
        self.layers, self.ops, self.max_layer = layers, ops, max_layer
        self.shuffle = shuffle
        self.bias_mode, self.round_mode, self.label = bias_mode, round_mode, label
        # P3.18 arm `gluefp`: which GLUE hops are executed in float32 instead of being
        # rounded to int8 at that op's own `out_sa`. `norm` covers NORMSILU/NORM/SILU,
        # `add` the residual adds. Convs always stay int8 — that is the artifact.
        self.glue_fp = set(glue_fp)
        # P3.18 arm `convoutfp`: the conv still computes its taps and requant EXACTLY,
        # but the result is kept as the fp32 value instead of being rounded to int8 at
        # SA_out. Prices the conv's OUTPUT grid, which the glue arms cannot: they all
        # still lose whatever the output hop lost.
        self.conv_out_fp = conv_out_fp
        # P3.18 instrumentation: the one-hop rounding cost of EVERY glue op, measured at
        # the walk's own input (cos of the op's fp32 result vs its own requantised codes).
        # It prices each op without needing a reference tensor at that site, so the
        # `wire` arm's 101-op chain gets a per-op number instead of one end figure.
        self.hops: list[dict] = []
        self.last_gather_hop: float | None = None
        # P3.16 arms: `add_fp` = conv indices whose FOLLOWING residual add is done in
        # float32 (the int8 hop at the add is removed; the next quantisation, the
        # scale-invariant `norm_out`, then does the only rounding). `cout_pc` = give
        # the 96->3 `conv_out` three per-channel output scales instead of one.
        self.add_fp = set(add_fp)
        self.cout_pc = cout_pc
        # P3.17 arm `oracle`: every conv is fed the REFERENCE module's own input,
        # quantised at the program's `SA_in`, instead of the walk's accumulated codes.
        # Its output cosine then measures the site itself (weights, output scale,
        # per-tap bias/rounding) with the upstream error removed.
        self.oracle = oracle
        self.oracle_in: dict[str, tuple] = {}         # name -> (path, shape)
        self.chan_sa: dict[int, np.ndarray] = {}      # ai -> per-channel sa_out
        self.sain_override: dict[int, float] = {}     # ai -> a re-calibrated input scale
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
        self.sa_vec: np.ndarray | None = None    # per-channel sa_out after a `cout_pc` conv
        self.last_conv: int | None = None
        self.adds: list[dict] = []               # every residual add's scale bookkeeping
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
        dot = n1 = n2 = 0.0
        for t in range(fin):
            if t == 0:
                srcf, half = 0, 0
            else:
                srcf, half = (t + 1) // 2, (t + 1) % 2
            if srcf >= act.shape[0]:
                continue
            sub = act[srcf][half * cin:(half + 1) * cin][:, ih][:, :, iw]
            if g == 1.0 and L["wiring"] in (0, 1, 2, 5) and act.dtype == np.int8:
                inb[t] = sub
                q = sub
                ideal = sub.astype(np.float32)
            else:
                ideal = sub.astype(np.float32) * np.float32(g)
                q, _ = q_clamp(ideal)
                inb[t] = q
            # P3.18: the one-hop cost of BUILDING this conv's input — the exact fp32
            # resample vs the int8 codes the conv actually consumes (the requant ratio
            # `sa_prev/sa_in` is the whole cost; the mapping itself is P3.14's).
            qf = q.astype(np.float32)
            dot += float(np.sum(ideal * qf, dtype=np.float64))
            n1 += float(np.sum(ideal * ideal, dtype=np.float64))
            n2 += float(np.sum(qf * qf, dtype=np.float64))
        self.last_gather_hop = (None if min(n1, n2) <= 0 else
                                float(1.0 - dot / max(math.sqrt(n1 * n2), 1e-30)))
        self.last_gather_g = float(g)
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
        dot = n1 = n2 = 0.0
        for t in range(fin):
            srcf = t // 2 if tshuf else t
            if srcf >= act.shape[0]:
                continue                              # device: memset 0
            half = (t % 2) * cin if tshuf else 0
            sub = act[srcf][half:half + cin][:, ih][:, :, iw]
            if L["wiring"] == 0 and g == 1.0 and act.dtype == np.int8:
                inb[t] = sub
                q = sub
                ideal = sub.astype(np.float32)
            else:
                ideal = sub.astype(np.float32) * np.float32(g)
                q, _ = q_clamp(ideal)
                inb[t] = q
            qf = q.astype(np.float32)                # P3.18: the input build's own hop
            dot += float(np.sum(ideal * qf, dtype=np.float64))
            n1 += float(np.sum(ideal * ideal, dtype=np.float64))
            n2 += float(np.sum(qf * qf, dtype=np.float64))
        self.last_gather_hop = (None if min(n1, n2) <= 0 else
                                float(1.0 - dot / max(math.sqrt(n1 * n2), 1e-30)))
        self.last_gather_g = float(g)
        return inb

    def conv(self, li: int) -> dict:
        L = self.layers[li]
        ai = L["ai"]
        self.last_gather_hop = None
        self.last_gather_g = None
        sa_in = float(self.sain_override.get(ai, self.SAIN[ai]))
        sa_out = float(self.SAOUT[ai])
        sc = self.SC[L["sc_off"]:L["sc_off"] + L["cout"]]
        bi = self.BI[L["sc_off"]:L["sc_off"] + L["cout"]]
        # per-output-channel scale for the 96->3 `conv_out` (P3.16 arm `coutpc`): the
        # artifact has one `sa_out` for all 3 channels, so a per-channel arm changes
        # both the requant and the dequant. `sc` is ALREADY per output channel.
        pc = self.cout_pc and ai in self.chan_sa
        if pc:
            sa_vec = np.asarray(self.chan_sa[ai], dtype=np.float64)
            assert sa_vec.size == L["cout"], f"{L['name']}: {sa_vec.size} scales for {L['cout']} ch"
            req = (sa_in * sc / sa_vec).reshape(-1, 1, 1).astype(np.float64)
            out_sa_use = float(sa_vec.mean())
        else:
            sa_vec = None
            req = (sa_in * sc / sa_out).reshape(-1, 1, 1).astype(np.float64)
            out_sa_use = sa_out
        # the C++ builds the INT32 bias in ACCUMULATOR units, then lets CONV_2D
        # requant: out = (acc + bias_int) * sa_in*sc[c] / sa_out
        bi_int = np.trunc(bi / (sa_in * sc)).astype(np.int64)
        W = self.weights(li)
        oi = self.oracle_in.get(L["name"]) if self.oracle else None
        if oi is not None:
            path, shape = oi
            want = (L["frames"], L["cin"], L["h"], L["w"])
            if tuple(shape) != want:
                raise SystemExit(f"{L['name']}: reference input {tuple(shape)} != program "
                                 f"{want} — the oracle arm cannot substitute here")
            inb = np.fromfile(path, dtype=np.int8).reshape(shape)
        else:
            inb = self.gather(L, sa_in)
        pad = (L["taps"][0]["kh"] // 2, L["taps"][0]["kw"] // 2)
        # exact integer accumulation: keep a partial sum below 2^24 in float32
        per_term = 127 * 127 * (2 * pad[0] + 1) * (2 * pad[1] + 1)
        step = max(1, int((2 ** 24) // max(1, per_term)))
        outb = np.zeros((L["frames"], L["cout"], L["h"], L["w"]), dtype=np.int8)
        outf = (np.zeros((L["frames"], L["cout"], L["h"], L["w"]), dtype=np.float32)
                if self.conv_out_fp else None)
        clamps = 0
        tap0_skipped = 0
        for f in range(L["frames"]):
            acc = np.zeros((L["cout"], L["h"], L["w"]), dtype=np.int32)
            applied = [(ti, f - T["delta"]) for ti, T in enumerate(L["taps"])
                       if 0 <= f - T["delta"] < L["fin"]]
            if applied and applied[0][0] != 0:
                tap0_skipped += 1
            bias_acc = np.zeros((L["cout"], L["h"], L["w"]), dtype=np.int32)
            for k, (ti, src) in enumerate(applied):
                x = torch.from_numpy(inb[src].astype(np.float32))
                acc_t = np.zeros((L["cout"], L["h"], L["w"]), dtype=np.int32)
                for c0 in range(0, L["cin"], step):
                    c1 = min(L["cin"], c0 + step)
                    wch = torch.from_numpy(W[ti][:, c0:c1].astype(np.float32))
                    part = F.conv2d(x[None, c0:c1], wch, padding=pad)[0]
                    acc_t += torch.round(part).to(torch.int32).numpy()
                # ---- the accumulating-scheme arms (P3.15). `wire` = the C++ as built:
                # every tap's graph carries the same INT32 bias, and each tap is
                # requantised+clamped to int8 before the sum. `bias1` gives the bias to
                # the first APPLIED tap only (tap 0's graph); `noround` keeps the bias on
                # every tap but sums in int32 and requants once; `exact` does both.
                b = 1 if (self.bias_mode == "per_tap" or k == 0) else 0
                if self.round_mode == "per_tap":
                    v = (acc_t.astype(np.float64) + b * bi_int.reshape(-1, 1, 1)) * req
                    q, n = q_clamp(v)
                    clamps += n
                    acc += q.reshape(L["cout"], L["h"], L["w"]).astype(np.int32)
                else:
                    acc += acc_t
                    if b:
                        bias_acc += bi_int.reshape(-1, 1, 1)
            if self.conv_out_fp:
                # No int8 hop at the conv's output: the value is kept as fp32 in value
                # units. The CLAMP is kept (it is the artifact's deliberate range, 127
                # codes) — only the rounding is removed, so this arm prices the grid's
                # step and nothing else. NB `acc` is already in CODE units in per_tap
                # mode (each tap was rounded to int8 before the sum).
                if self.round_mode == "per_tap":
                    qf = np.clip(acc, -127, 127).astype(np.float64)
                else:
                    qf = np.clip((acc + bias_acc).astype(np.float64) * req, -127.0, 127.0)
                outf[f] = (qf * out_sa_use).astype(np.float32)
            else:
                if self.round_mode == "per_tap":
                    q2, n2 = q_clamp(acc)
                else:
                    q2, n2 = q_clamp((acc + bias_acc).astype(np.float64) * req)
                clamps += n2
                outb[f] = q2.reshape(L["cout"], L["h"], L["w"])
        self.clamps += clamps
        if self.conv_out_fp:
            self.act = outf
            self.sa = 1.0
            self.sa_vec = None
        else:
            self.act = outb
            self.sa = out_sa_use
            self.sa_vec = sa_vec
        self.last_conv = li
        self.hist.append((outb, out_sa_use, li))
        self.hist = self.hist[-3:]
        self.last_inb = inb
        self.last_inb_name = L["name"]
        r = dict(idx=li, ai=ai, name=L["name"], shape=list(outb.shape), sa_out=out_sa_use,
                 sa_in=float(sa_in), oracle=bool(oi is not None),
                 clamp=clamps / outb.size, scheme=self.label, tap0_skipped=tap0_skipped,
                 gather_hop=self.last_gather_hop, gather_g=self.last_gather_g,
                 conv_out_fp=bool(self.conv_out_fp))
        if sa_vec is not None:
            r.update(pc=True, sa_out_pc=[float(v) for v in sa_vec])
        self.rec.append(r)
        return r

    # ---------------------------------------------------------------- the float ops
    def normsilu(self, goff: int, C: int, gam_scale: float, out_sa: float,
                 do_norm: bool, do_silu: bool, kind: str = "NORMSILU") -> None:
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
        if "norm" in self.glue_fp:
            # P3.18 `gluefp`: no int8 hop here — the float result IS the state (scale
            # 1.0), so the next requant is the conv's input build and nothing else.
            self.act = y
            self.sa = 1.0
            self.sa_vec = None
            self.hops.append(dict(kind=kind, after_conv=self.last_conv, C=int(C),
                                  sa_in=float(sain), out_sa=float(out_sa),
                                  hop=None, clamp=None, fp=True))
            return
        q, n = q_clamp(y / np.float32(out_sa))
        self.clamps += n
        # P3.18: this op's own one-hop rounding cost, at the input it was handed.
        qd = q.astype(np.float32) * np.float32(out_sa)
        dot = float(np.sum(y * qd, dtype=np.float64))
        n1 = float(np.sum(y * y, dtype=np.float64))
        n2 = float(np.sum(qd * qd, dtype=np.float64))
        self.hops.append(dict(kind=kind, after_conv=self.last_conv, C=int(C),
                              sa_in=float(sain), out_sa=float(out_sa),
                              hop=float(1.0 - dot / max(math.sqrt(n1 * n2), 1e-30)),
                              clamp=float(n) / q.size, fp=False, elements=int(q.size)))
        self.act = q.reshape(a.shape)
        self.sa = float(out_sa)
        self.sa_vec = None

    def add(self, slot: int, out_sa: float) -> None:
        s, s_sa = self.slot[slot & 1]
        assert s.shape == self.act.shape, f"ADD act {self.act.shape} vs slot {s.shape}"
        # what this add does to the two operands' grids: `ratio` > 1 means the sum is
        # written on a COARSER grid than the finer operand, i.e. that operand's
        # LSB is multiplied by `ratio` at the add.
        rec = dict(after_conv=self.last_conv, sa_act=float(self.sa), sa_slot=float(s_sa),
                   out_sa=float(out_sa),
                   ratio=float(out_sa / max(min(self.sa, s_sa), 1e-30)))
        if self.last_conv in self.add_fp:
            # P3.16 arm `addfp`: no int8 hop at the add. The sum stays float32 and the
            # next op (the scale-invariant `norm_out`) does the only rounding.
            self.act = (self.act.astype(np.float32) * np.float32(self.sa)
                        + s.astype(np.float32) * np.float32(s_sa))
            self.sa = 1.0
            rec.update(fp=True, clamp=0.0, hop=None)
        elif "add" in self.glue_fp:
            # P3.18 `gluefp`: the same, for every residual add.
            self.act = (self.act.astype(np.float32) * np.float32(self.sa)
                        + s.astype(np.float32) * np.float32(s_sa))
            self.sa = 1.0
            rec.update(fp=True, clamp=0.0, hop=None)
        else:
            v = (self.act.astype(np.float64) * self.sa + s.astype(np.float64) * s_sa) / out_sa
            q, n = q_clamp(v)
            self.clamps += n
            qd = q.astype(np.float32) * np.float32(out_sa)
            vf = v.astype(np.float32) * np.float32(out_sa)
            dot = float(np.sum(vf * qd, dtype=np.float64))
            n1 = float(np.sum(vf * vf, dtype=np.float64))
            n2 = float(np.sum(qd * qd, dtype=np.float64))
            rec.update(fp=False, clamp=n / q.size, elements=int(q.size),
                       hop=float(1.0 - dot / max(math.sqrt(n1 * n2), 1e-30)))
            self.act = q.reshape(self.act.shape)
            self.sa = float(out_sa)
        self.sa_vec = None
        self.adds.append(rec)
        self.hops.append(dict(kind="ADD", after_conv=self.last_conv,
                              sa_in=rec["sa_act"], out_sa=float(out_sa),
                              ratio=rec["ratio"], hop=rec.get("hop"),
                              clamp=rec.get("clamp"), fp=rec["fp"]))

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
                self.normsilu(op[1], op[2], op[3], op[4], True, k == "NORMSILU", kind=k)
            elif k == "SILU":
                self.normsilu(0, self.act.shape[1], 1.0, op[1], False, True, kind="SILU")
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
                    sa_in: dict[str, float], tmp: Path, bypass_pqc: bool = True,
                    want_pqc_z: bool = False) -> tuple[dict, torch.Tensor, np.ndarray | None]:
    """fp32 decode with the program's deviations (attention skipped, and by default
    `post_quant_conv` bypassed), each layer module's output concatenated over the
    decoder's frame chunks and written as int8 codes at that layer's own `SA_out`.

    `bypass_pqc=False` runs the op the artifact does not carry (P3.15 candidate 3), so
    the trace is the "correct program" target; `want_pqc_z=True` also returns the
    op's own output as [f, c, h, w], which is the latent a device program would have
    to consume if the op were folded in.
    """
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(str(VAE_DIR), torch_dtype=torch.float32).eval()
    calls: dict[str, list] = {n: [] for n in names}
    # P3.17: EVERY conv's own input, not just `resample.1`'s. The oracle-input arm
    # needs the tensor each module actually consumed, in [frames, C, H, W] order.
    ins: dict[str, list] = {n: [] for n in names}
    # `upsample3d`'s `time_conv` is the one module the streaming decode does NOT run
    # the way the program does: `WanResample.forward` skips it for the first latent
    # frame (it sets a "Rep" sentinel and replicates that frame) and then calls it one
    # frame at a time with a zeroed causal cache, so its hooked input is T-1 frames and
    # its hooked output T-1 frames — which is why layers 11/20 read SHAPE MISMATCH in
    # every arm to date. Its real input is the upsample block's input x, so hook the
    # PARENT for those two and rebuild the module's own full-sequence output after the
    # decode. This is the only substitution in the tool, and it is logged.
    par = {n: n[: -len(".time_conv")] for n in names if n.endswith(".time_conv")}
    pins: dict[str, list] = {n: [] for n in par}
    hooks = []

    def _mk_in_hook(n):
        def h(m, i):
            # Quantise in the hook: 36 convs hold ~1.4 G elements of input, so keeping
            # float32 clones would cost ~5.6 GB. int8 at SA_in is exactly what the
            # conv consumes anyway.
            q = torch.clamp(torch.round(i[0].detach() / float(sa_in[n])),
                            -127.0, 127.0).to(torch.int8)
            if q.dim() == 5:              # conv3d [b, c, t, h, w] -> [t, c, h, w]
                q = q[0].permute(1, 0, 2, 3)
            ins[n].append(q)
        return h

    def _mk_par_hook(n):
        def h(m, i):
            x = i[0].detach()
            pins[n].append(x[0].permute(1, 0, 2, 3).contiguous())   # [t, c, h, w] fp32
        return h

    for n in names:
        mod = vae.get_submodule(n)
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, n=n: calls[n].append(o.detach().clone())))
        hooks.append(mod.register_forward_pre_hook(_mk_in_hook(n)))
    for n, p in par.items():
        hooks.append(vae.get_submodule(p).register_forward_pre_hook(_mk_par_hook(n)))
    hooks.append(vae.decoder.mid_block.attentions[0].register_forward_hook(lambda m, i, o: i[0]))
    if bypass_pqc:
        hooks.append(vae.post_quant_conv.register_forward_hook(lambda m, i, o: i[0]))
    pqc_out = None
    with torch.no_grad():
        if want_pqc_z and not bypass_pqc:
            pqc_out = vae.post_quant_conv(z).detach().clone()      # [1, C, F, H, W]
        out = vae.decode(z).sample
    for h in hooks:
        h.remove()

    # ---- P3.17: rebuild `upsample3d`'s time_conv input AND target in whole-sequence form.
    # The streaming reference ran it on T-1 single frames (frame 0 replicated, not convolved),
    # so both its hooked input and its hooked output are short by one frame. Its real input is
    # the upsample block's input; running the same module on that input is the target the
    # program's 3-frame conv is trying to hit. Only layers 11/20 are affected.
    rebuilt = {}
    for n in par:
        if not pins[n]:
            continue
        xin = torch.cat(pins[n], dim=0)                        # [T, C, H, W] fp32
        mod = vae.get_submodule(n)
        with torch.no_grad():
            yout = mod(xin.permute(1, 0, 2, 3).unsqueeze(0))    # [1, 2C, T, H, W]
        yout = yout[0].permute(1, 0, 2, 3).contiguous()        # [T, 2C, H, W]
        st_in = int(sum(t.shape[0] for t in ins[n]))
        st_out = int(sum(t.shape[2] for t in calls[n])) if calls[n] and calls[n][0].dim() == 5 \
            else int(sum(t.shape[0] for t in calls[n]))
        ins[n] = [torch.clamp(torch.round(xin / float(sa_in[n])), -127.0, 127.0).to(torch.int8)]
        calls[n] = [yout]
        rebuilt[n] = dict(streaming_in_frames=st_in, streaming_out_frames=st_out,
                          whole_in_frames=int(xin.shape[0]), whole_out_frames=int(yout.shape[0]),
                          note="the streaming decode skips `time_conv` on the first latent frame "
                               "(replicates it) and calls it one frame at a time with a zeroed "
                               "causal cache, so its own input/output are T-1 frames. Rebuilt from "
                               "the upsample block's input — same module, whole-sequence form.")
        log(f"time_conv {n.split('decoder.')[-1]}: streaming in/out {st_in}/{st_out} frames -> "
            f"whole-sequence {xin.shape[0]}/{yout.shape[0]}")
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
                     clamp=float((cat / sa).abs().gt(127).float().mean()),
                     # per-output-channel range of the reference's OWN fp32 module output.
                     # The artifact carries ONE `sa_out` per layer, so this is the
                     # measurement that says what a per-channel scale would be worth
                     # (P3.16, the `conv_out` 96->3 leg).
                     chmax=[float(v) for v in cat.abs().amax(dim=(0, 2, 3))])
        if ins.get(n):
            ic = torch.cat(ins[n], dim=0)              # [frames, C, H, W] in output-frame order
            p = tmp / f"{n}.in.i8"
            p.write_bytes(ic.numpy().tobytes())        # already int8 at sa_in
            tr[n]["in_path"] = str(p)
            tr[n]["in_sa"] = float(sa_in[n])
            tr[n]["in_shape"] = list(ic.shape)
            tr[n]["in_clamp"] = float(ic.abs().eq(127).float().mean())
            ins[n] = []                                # free the int8 chunks (~1.4 GB total)
    pqc_np = None
    if pqc_out is not None:
        pqc_np = pqc_out.squeeze(0).permute(1, 0, 2, 3).contiguous().numpy()   # [f,c,h,w]
    tr["_rebuilt_time_conv"] = rebuilt
    return tr, out, pqc_np


def cosine_codes(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def _metrics(dev: torch.Tensor, ref: torch.Tensor) -> dict:
    """`vae_psnr.py`'s metric block (Gate 1's data_range 2.0), so the numbers are comparable."""
    if tuple(dev.shape) != tuple(ref.shape):
        return {"shape_mismatch": [list(dev.shape), list(ref.shape)]}
    d = dev - ref
    rmse = float(torch.sqrt(torch.mean(d ** 2)))
    rms_ref = float(torch.sqrt(torch.mean(ref ** 2)))
    peak = float(ref.abs().max())
    rmse_pk = float(torch.sqrt(torch.mean(d ** 2)))
    return {
        "psnr_db": float("inf") if rmse == 0 else 20.0 * math.log10(2.0 / rmse),
        "psnr_db_datapeak": (float("inf") if rmse_pk == 0 else
                             20.0 * math.log10(peak / rmse_pk)) if peak > 0 else None,
        "cosine": float(F.cosine_similarity(dev.flatten(), ref.flatten(), dim=0)),
        "rmse": rmse,
        "rel_rmse": rmse / max(rms_ref, 1e-12),
        "ref_absmax": peak,
        "ref_rms": rms_ref,
    }


def final_metrics(act_i8: np.ndarray, sa: float, ref: torch.Tensor) -> dict:
    """The walk's final codes, dequantised at its own scale, vs an fp32 decode."""
    dev = (torch.from_numpy(act_i8.astype(np.float32) * np.float32(sa))
           .permute(1, 0, 2, 3).unsqueeze(0).contiguous())
    return _metrics(dev, ref)


def final_metrics_pc(act_i8: np.ndarray, sa_vec, ref: torch.Tensor) -> dict:
    """Same metric with each output channel dequantised at its OWN scale (P3.16)."""
    m = torch.from_numpy(np.asarray(sa_vec, dtype=np.float32)).reshape(1, -1, 1, 1)
    dev = (torch.from_numpy(act_i8.astype(np.float32)) * m
           ).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    return _metrics(dev, ref)


def final_metrics_diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Two fp32 decodes: measures what an op changes at all, with no quantisation in it."""
    return _metrics(a, b)



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wire", default=str(WIRE))
    ap.add_argument("--max-layer", type=int, default=10 ** 6)
    ap.add_argument("--latent", default=None, help="a dump_<ts> dir (default: newest)")
    ap.add_argument("--device", default=None,
                    help="a dump_<ts> dir whose vae_dump.bin is the device's final decode")
    ap.add_argument("--shuffle", default="both", choices=["wire", "shift", "both"],
                    help="temporal-upsample gather: `wire` = npu_block.cpp's (t//2, t%%2); "
                         "`shift` = the mapping measured off the reference's own conv input; "
                         "`both` measures them side by side")
    ap.add_argument("--schemes", default=None,
                    help="score the accumulating-scheme arms instead of the shuffle arms, at "
                         "the measured `shift` gather: `all` or a comma list of "
                         "wire,bias1,noround,exact (bias1 = INT32 bias on the first applied tap "
                         "only; noround = taps summed in int32 and requantised once)")
    ap.add_argument("--pqc", action="store_true",
                    help="also score the MISSING `post_quant_conv` (16->16): runs the op under "
                         "the reference, feeds the walk the op's quantised output, and measures "
                         "both programme variants against the same fp32 target")
    ap.add_argument("--layer34", action="store_true",
                    help="score the two layer-34 candidates (P3.16) at the measured `shift` "
                         "gather: `add34` = the last up_block's residual add done in float32 "
                         "(no int8 hop; `norm_out` then does the only rounding), `coutpc` = "
                         "3 per-channel output scales for the 96->3 `conv_out`, `both34` = both")
    ap.add_argument("--oracle", action="store_true",
                    help="P3.17: run the `wire` arm and an `oracle` arm in which every conv is fed "
                         "the REFERENCE module's own input, quantised at the program's SA_in, so a "
                         "layer's cosine separates inherited error (input already wrong) from "
                         "intrinsic error (the conv/weights/scale at that site)")
    ap.add_argument("--glue", action="store_true",
                    help="P3.18a: price the GLUE hops, one class at a time. Arms: `wire` (validator), "
                         "`gluefp` (every NORMSILU/NORM/SILU and every residual ADD executed in "
                         "float32 — no int8 hop in the glue at all), `normfp` (norms only), `addfp` "
                         "(adds only). Every arm also records the ONE-HOP cost of each glue op and "
                         "of each conv's input build, measured at the input that op was handed")
    ap.add_argument("--convout", action="store_true",
                    help="P3.18b: price the conv's OUTPUT grid. Arms: `wire` (validator) and "
                         "`convoutfp`, where every conv keeps its exact requant value in float32 "
                         "instead of rounding it to int8 at SA_out (the taps, their per-tap int8 "
                         "rounding and the input build are unchanged)")
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
    tr, out, _ = reference_trace(z, names, sa_out, sa_in_d, tmp)
    log(f"fp32 reference in {time.time() - t0:.1f} s -> {tuple(out.shape)}")

    # ------------------------------------------------------------------ the arm matrix
    chan_sa: dict[int, np.ndarray] = {}          # ai -> per-channel sa_out (P3.16 `coutpc`)
    ARMS = {                     # label: (shuffle, bias placement, rounding granularity)
        "wire":    ("shift", "per_tap", "per_tap"),
        "bias1":   ("shift", "once",    "per_tap"),
        "noround": ("shift", "per_tap", "once"),
        "exact":   ("shift", "once",    "once"),
    }
    if args.schemes:
        keys = list(ARMS) if args.schemes == "all" else args.schemes.split(",")
        for k in keys:
            if k not in ARMS:
                raise SystemExit(f"unknown arm {k!r}; choose from {list(ARMS)}")
        arms = [(k, *ARMS[k], {}) for k in keys]
    elif args.layer34:
        # P3.16 — the two named layer-34 candidates, one at a time and together.
        L35 = next((L for L in layers if L["idx"] == 35), None)
        if L35 is None or L35["name"] not in tr:
            raise SystemExit("--layer34 needs a walk through conv_out; drop --max-layer")
        ARMS34 = {
            "wire":   {},
            "add34":  dict(add_fp=(34,)),
            "coutpc": dict(cout_pc=True),
            "both34": dict(add_fp=(34,), cout_pc=True),
        }
        arms = [(k, "shift", "per_tap", "per_tap", v) for k, v in ARMS34.items()]
        # per-channel output scales for the 96->3 `conv_out`, from the reference's OWN
        # per-channel max (the same max/127 convention `calibrate.py` uses).
        L35 = next(L for L in layers if L["idx"] == 35)
        ch = tr[L35["name"]]["chmax"]
        chan_sa[L35["ai"]] = np.asarray(ch, dtype=np.float64) / 127.0
        log(f"conv_out per-channel max {['%.4f' % v for v in ch]} -> sa "
            f"{['%.6g' % v for v in chan_sa[L35['ai']]]} "
            f"(single sa {float(sim.SAOUT[L35['ai']]):.6g}, "
            f"gain {max(ch) / (min(ch) or 1):.2f}x between channels)")
    elif args.oracle:
        # P3.17 — the discrimination P3.16 named: same walk, same scheme, only the
        # conv's INPUT changes.
        arms = [("wire", "shift", "per_tap", "per_tap", {}),
                ("oracle", "shift", "per_tap", "per_tap", dict(oracle=True))]
    elif args.glue:
        # P3.18 — the glue hops, class by class. The convs are identical in all four
        # arms; only which GLUE ops round to int8 changes, so the difference in the
        # end product is that op class's contribution to the walk.
        GARMS = {
            "wire":   (),                 # validator: must reproduce P3.17 to the digit
            "gluefp": ("norm", "add"),    # no int8 hop anywhere in the glue
            "normfp": ("norm",),          # norm+SiLU hops removed, adds still int8
            "addfp":  ("add",),           # residual adds in float32, norms still int8
        }
        arms = [(k, "shift", "per_tap", "per_tap", dict(glue_fp=v)) for k, v in GARMS.items()]
    elif args.convout:
        # P3.18b — the conv's output grid, alone. Everything the conv computes is
        # identical up to the final requant; only whether that value is rounded to int8
        # at SA_out changes.
        arms = [("wire", "shift", "per_tap", "per_tap", {}),
                ("convoutfp", "shift", "per_tap", "per_tap", dict(conv_out_fp=True))]
    else:
        modes = ["wire", "shift"] if args.shuffle == "both" else [args.shuffle]
        arms = [(m, m, "per_tap", "per_tap", {}) for m in modes]

    results: dict[str, dict] = {}
    finals: dict[str, tuple[np.ndarray, float]] = {}

    def run_arm(label: str, shuffle: str, bias_mode: str, round_mode: str,
                lat_use: np.ndarray, lat_sa: float, tr_use: dict, out_use: torch.Tensor,
                override: dict[int, float] | None = None, add_fp: tuple = (),
                cout_pc: bool = False, oracle: bool = False, glue_fp: tuple = (),
                conv_out_fp: bool = False) -> dict:
        log(f"--- arm {label}: gather={shuffle} bias={bias_mode} round={round_mode}"
            + (f" add_fp={add_fp}" if add_fp else "") + (" cout_pc=1" if cout_pc else "")
            + (f" glue_fp={','.join(glue_fp)}" if glue_fp else "")
            + (" convout_fp=1" if conv_out_fp else "")
            + (" oracle_input=1" if oracle else ""))
        s = Sim(layers, ops, args.max_layer, shuffle=shuffle, bias_mode=bias_mode,
                round_mode=round_mode, label=label, add_fp=add_fp, cout_pc=cout_pc,
                oracle=oracle, glue_fp=glue_fp, conv_out_fp=conv_out_fp)
        if oracle:
            s.oracle_in = {n: (v["in_path"], tuple(v["in_shape"]))
                           for n, v in tr_use.items() if v.get("in_path")}
            miss = [L["name"] for L in layers if L["idx"] <= args.max_layer
                    and L["name"] not in s.oracle_in]
            if miss:
                raise SystemExit(f"oracle: no reference input for {miss}")
            log(f"    oracle inputs: {len(s.oracle_in)} convs, "
                f"{sum(np.prod(sh) for _p, sh in s.oracle_in.values()) / 1e6:.1f} M codes")
        if cout_pc:
            s.chan_sa = chan_sa
        if override:
            s.sain_override = override
        rows, first_bad = [], None
        orig_conv = s.conv

        def conv_and_cmp(li: int) -> dict:
            nonlocal first_bad
            r = orig_conv(li)
            ref = tr_use.get(r["name"])
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
                    if s.sa_vec is not None:
                        # the codes are on a different (per-channel) grid, so score them
                        # DEQUANTISED as well — the code-space cosine is not the metric
                        rd = (torch.from_numpy(ref_codes.astype(np.float32))
                              * np.float32(ref["sa"])).flatten()
                        sd = (torch.from_numpy(s.act.astype(np.float32))
                              * torch.from_numpy(np.asarray(s.sa_vec, dtype=np.float32))
                              .reshape(1, -1, 1, 1)).flatten()
                        r["cosine_deq"] = float(F.cosine_similarity(sd, rd, dim=0))
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
        t0 = time.time()
        s.run(lat_use, lat_sa)
        fin = final_metrics(s.act, s.sa, out_use)
        fin_pc = final_metrics_pc(s.act, s.sa_vec, out_use) if s.sa_vec is not None else None
        finals[label] = (s.act.copy(), float(s.sa))
        res = dict(convs=rows, first_divergence=first_bad, total_clamps=int(s.clamps),
                   final=fin, final_pc=fin_pc, adds=s.adds, hops=s.hops,
                   walk_s=time.time() - t0, bias=bias_mode, round=round_mode,
                   gather=shuffle, add_fp=list(add_fp), cout_pc=bool(cout_pc),
                   glue_fp=list(glue_fp), conv_out_fp=bool(conv_out_fp),
                   worst=min((r for r in rows if r.get("cosine") is not None),
                             key=lambda r: r["cosine"], default={}))
        if fin_pc is not None:
            log(f"{label}: final (per-channel dequant) PSNR(2.0) {fin_pc['psnr_db']:.2f} dB · "
                f"cosine {fin_pc['cosine']:.6f}")
        log(f"{label}: final PSNR(2.0) {fin.get('psnr_db', float('nan')):.2f} dB · cosine "
            f"{fin.get('cosine', float('nan')):.6f} · worst layer "
            f"{res['worst'].get('idx')} {res['worst'].get('cosine', float('nan')):.6f}")
        return res

    for label, shuffle, bm, rm, extra in arms:
        results[label] = run_arm(label, shuffle, bm, rm, lat, lmeta["sa"], tr, out, **extra)

    # ------------------------------------------------------------- candidate 3: the missing op
    pqc_doc = None
    if args.pqc:
        log("--- post_quant_conv: reference WITH the op (fp32 target) + the walk that consumes it")
        tmp2 = Path(tempfile.mkdtemp(prefix="vae_sim_pqc_"))
        t0 = time.time()
        tr2, out2, zpq = reference_trace(z, names, sa_out, sa_in_d, tmp2,
                                        bypass_pqc=False, want_pqc_z=True)
        log(f"fp32 reference (post_quant_conv ON) in {time.time() - t0:.1f} s")
        d_pqc = final_metrics_diff(out2, out)        # how much the op changes the decode at all
        sa_pq = float(np.abs(zpq).max()) / 127.0
        lat_pq, n_sat = q_clamp(zpq / sa_pq)
        log(f"post_quant_conv out {zpq.shape} range {zpq.min():.4f}..{zpq.max():.4f} "
            f"-> quantised at sa {sa_pq:.6g} ({n_sat} codes clamped)")
        base = "exact" if "exact" in results else arms[0][0]
        i0 = next(L["ai"] for L in layers if L["idx"] == 0)
        pqc_label = f"{base}+pqc"
        results[pqc_label] = run_arm(pqc_label, "shift", "once", "once", lat_pq, sa_pq, tr2, out2,
                                    override={i0: sa_pq})
        # both programme variants against the SAME target: the fp32 decode with the op in it
        same_target = {k: final_metrics(finals[k][0], finals[k][1], out2)
                       for k in (base, pqc_label) if k in finals}
        pqc_doc = dict(sa=sa_pq, clamped_codes=int(n_sat), fp32_vs_fp32=d_pqc,
                       against_pqc_target=same_target, base_arm=base,
                       op="post_quant_conv 16->16, absent from artifacts/vae (36 layers = decoder only)")
        log(f"post_quant_conv is worth: {same_target[base]['psnr_db']:.2f} dB -> "
            f"{same_target[pqc_label]['psnr_db']:.2f} dB against the same fp32 target "
            f"(cosine {same_target[base]['cosine']:.6f} -> {same_target[pqc_label]['cosine']:.6f})")
        shutil.rmtree(tmp2, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------- P3.17: inherited vs intrinsic error
    oracle_doc = None
    if args.oracle and "oracle" in results and "wire" in results:
        wc, oc = results["wire"]["convs"], results["oracle"]["convs"]
        rows = []
        for a, b in zip(wc, oc):
            ci = a.get("cos_gather")
            rows.append(dict(idx=a["idx"], ai=a["ai"], name=a["name"], sa_in=b.get("sa_in"),
                             in_cos=ci, wire_cos=a.get("cosine"), oracle_cos=b.get("cosine"),
                             inherited=None if ci is None else 1.0 - ci,
                             intrinsic=None if b.get("cosine") is None else 1.0 - b["cosine"]))
        intr = [r["intrinsic"] for r in rows if r["intrinsic"] is not None]
        inh = [r["inherited"] for r in rows if r["inherited"] is not None]
        sites = [r for r in rows if r["oracle_cos"] is not None]
        oracle_doc = dict(
            rows=rows, n=len(rows),
            mean_intrinsic=float(np.mean(intr)) if intr else None,
            median_intrinsic=float(np.median(intr)) if intr else None,
            max_intrinsic=float(np.max(intr)) if intr else None,
            mean_inherited=float(np.mean(inh)) if inh else None,
            sites_intrinsic_gt_1pct=int(sum(1 for v in intr if v > 0.01)),
            sites_intrinsic_gt_2pct=int(sum(1 for v in intr if v > 0.02)),
            worst_site=min(sites, key=lambda r: r["oracle_cos"]) if sites else {},
            layer34=next((r for r in rows if r["idx"] == 34), None),
            time_conv_rebuilt=tr.get("_rebuilt_time_conv"),
            note="`in_cos` = the wire arm's own conv input vs the reference module's input "
                 "quantised at the program's SA_in; `wire_cos` = the conv's output cosine "
                 "(= what P3.15/P3.16 report); `oracle_cos` = the SAME conv fed the reference's "
                 "own input. intrinsic = 1 - oracle_cos (the site's own loss on a perfect "
                 "input); inherited = 1 - in_cos (the error it was handed).")
        log(f"oracle: mean intrinsic loss {oracle_doc['mean_intrinsic'] * 100:.2f} pts, "
            f"mean inherited {oracle_doc['mean_inherited'] * 100:.2f} pts, "
            f"{oracle_doc['sites_intrinsic_gt_1pct']}/{len(intr)} sites > 1 pt intrinsic")

    # ------------------------------------------- P3.18: the glue hops, one at a time
    glue_doc = None
    if args.glue and "wire" in results:
        w = results["wire"]
        sites: list[dict] = []
        for r in w["convs"]:
            if r.get("gather_hop") is None:
                continue
            sites.append(dict(kind="INPUT-BUILD", after_conv=r["idx"], name=r["name"],
                              sa_in=r["sa_in"], sa_out=r["sa_out"], g=r.get("gather_g"),
                              hop=r["gather_hop"], clamp=None, fp=False))
        for h in w["hops"]:
            sites.append(dict(kind=h["kind"], after_conv=h["after_conv"],
                              name=next((L["name"] for L in layers
                                         if L["idx"] == h["after_conv"]), ""),
                              sa_in=h.get("sa_in"), sa_out=h.get("out_sa"),
                              g=h.get("ratio"), hop=h.get("hop"), clamp=h.get("clamp"),
                              fp=bool(h.get("fp"))))
        by: dict[str, dict] = {}
        for s in sites:
            b = by.setdefault(s["kind"], dict(n=0, hops=[], clamps=[], worst=None))
            if s["hop"] is None:
                continue
            b["n"] += 1
            b["hops"].append(s["hop"])
            if s.get("clamp") is not None:
                b["clamps"].append(s["clamp"])
            if b["worst"] is None or s["hop"] > b["worst"]["hop"]:
                b["worst"] = s
        summary = {k: dict(n=v["n"], sum_pt=sum(v["hops"]) * 100,
                           mean_pt=(sum(v["hops"]) / len(v["hops"]) * 100) if v["hops"] else None,
                           max_pt=max(v["hops"]) * 100 if v["hops"] else None,
                           worst_after_conv=(v["worst"] or {}).get("after_conv"),
                           mean_clamp_pct=(sum(v["clamps"]) / len(v["clamps"]) * 100)
                           if v["clamps"] else None)
                   for k, v in by.items()}
        glue_doc = dict(
            sites=sites, by_class=summary,
            arms={k: dict(final=results[k]["final"], glue_fp=results[k]["glue_fp"],
                          total_clamps=results[k]["total_clamps"],
                          worst=results[k].get("worst")) for k in results},
            note="One-hop cost of each op = 1 - cos(fp32 result of the op, that result "
                 "requantised to the op's own out_sa and dequantised again): what the int8 "
                 "hop at that op costs, priced AT THE INPUT THE WALK HANDED IT. `INPUT-BUILD` "
                 "= building a conv's input (temporal shuffle / 2x upsample + requant to that "
                 "conv's SA_in). The class sums are not additive in dB — they rank the classes "
                 "and locate the worst site; the arms are what price a class at the end product.")
        for k, v in summary.items():
            log(f"glue-hop {k:12s}: n={v['n']:3d} sum {v['sum_pt']:.2f} pt, mean "
                f"{v['mean_pt']:.3f} pt, max {v['max_pt']:.3f} pt (after conv "
                f"{v['worst_after_conv']})")

    ts = time.strftime("%Y%m%d_%H%M%S")
    labels = list(results)
    doc = dict(kind="vae_program_sim", timestamp=ts, wire=str(Path(args.wire).name),
               latent_dir=ddir.name, max_layer=args.max_layer, results=results,
               post_quant_conv=pqc_doc, oracle=oracle_doc, glue=glue_doc,
               note="cosine = the simulator's int8 codes vs the fp32 reference's own module "
                    "output quantised at the same SA_out; a collapse localizes the op between "
                    "the previous conv and this one. `final` = the walk's last codes "
                    "dequantised at their own scale vs the fp32 decode, PSNR on data_range 2.0 "
                    "as Gate 1/vae_psnr.py.")
    if args.layer34:
        L35 = next(L for L in layers if L["idx"] == 35)
        doc["layer34"] = dict(conv_out=L35["name"], conv_out_chmax=tr[L35["name"]]["chmax"],
                             conv_out_sa_single=float(sim.SAOUT[L35["ai"]]),
                             conv_out_sa_pc=[float(v) for v in chan_sa[L35["ai"]]],
                             add_grid=results[labels[0]]["adds"],
                             note="P3.16: `add34` = the residual add after conv 34 in float32; "
                                  "`coutpc` = three per-channel output scales for conv_out. "
                                  "`ratio` at an add = out_sa / min(operand sa)")
    js = OUT / f"{args.out}_{ts}.json"
    js.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    md = [f"# VAE program simulator — {ts}", "",
          f"wire `{Path(args.wire).name}` · latent `{ddir.name}` · max layer {args.max_layer}",
          "", "`wire` = the gather `npu_block.cpp` implements; `shift` = the mapping measured "
          "off the reference decoder's own conv input. Same latent, artifact and scales.", "",
          "## The walk's end product, per arm", "",
          "| arm | bias | rounding | final PSNR (2.0) | cosine | rel-RMSE | worst layer | clamp % |",
          "|---|---|---|---|---|---|---|---|"]
    for lb in labels:
        r = results[lb]
        fin = r.get("final", {})
        w = r.get("worst") or {}
        md.append(f"| **{lb}** | {r['bias']} | {r['round']} | "
                  + ("—" if "psnr_db" not in fin else f"{fin['psnr_db']:.2f} dB")
                  + " | " + ("—" if "cosine" not in fin else f"{fin['cosine']:.6f}")
                  + " | " + ("—" if "rel_rmse" not in fin else f"{fin['rel_rmse'] * 100:.2f} %")
                  + " | " + ("—" if not w else f"{w.get('idx')} ({w.get('cosine'):.6f})")
                  + f" | {max((c.get('clamp') or 0.0) for c in r['convs']) * 100:.4f} |")
    md += ["", "## Per layer, one cosine column per arm", "",
           "| layer | ai | module | shape | " + " | ".join(f"cos ({lb})" for lb in labels) + " |",
           "|---|---|---|---|" + "---|" * len(labels)]
    base_rows = results[labels[0]]["convs"]
    for i, r in enumerate(base_rows):
        cells = []
        for lb in labels:
            o = results[lb]["convs"][i] if i < len(results[lb]["convs"]) else None
            c = o.get("cosine") if o else None
            cells.append("—" if c is None else f"{c:.6f}")
        md.append(f"| {r['idx']} | {r['ai']} | `{r['name']}` | {r['shape']} | "
                  + " | ".join(cells) + " |")
    for lb in labels:
        fb = results[lb]["first_divergence"]
        md += ["", f"**{lb}**: first divergence " + (
            "none (all convs >= 0.99)" if not fb else
            f"layer {fb['idx']} (`{fb['name']}`) at cosine {fb['cosine']:.6f}")]
    if pqc_doc:
        st, d0 = pqc_doc["against_pqc_target"], pqc_doc["fp32_vs_fp32"]
        md += ["", "## Missing `post_quant_conv` (16→16), scored", "",
               f"Both programme variants against the SAME fp32 target — the decode with the op in "
               f"it (attention skipped, as the program skips it). Its output quantises at "
               f"`sa = {pqc_doc['sa']:.6g}` ({pqc_doc['clamped_codes']} clamped codes).", "",
               "| variant | PSNR (2.0) | cosine | rel-RMSE |", "|---|---|---|---|"]
        for k, v in st.items():
            md.append(f"| {k} | {v['psnr_db']:.2f} dB | {v['cosine']:.6f} | "
                      f"{v['rel_rmse'] * 100:.2f} % |")
        md += ["", f"fp32 vs fp32 (op on vs bypassed, no quantisation anywhere): "
               f"{d0['psnr_db']:.2f} dB / cosine {d0['cosine']:.6f} — i.e. the op changes the "
               f"decode by that much, which is the error the device walk carries for free while "
               f"the artifact has no `post_quant_conv`."]
    if args.layer34:
        L35 = next(L for L in layers if L["idx"] == 35)
        ch = tr[L35["name"]]["chmax"]
        md += ["", "## P3.16 — layer 34: the residual add's grid and the `conv_out` 96→3 shape", "",
               "Every residual add writes its sum on ONE grid (`post_add_sa`, calibrated from the",
               "post-add max). `ratio` = that grid's LSB over the finer operand's own LSB, so a",
               "`ratio` of 4 means the branch's quantisation step is multiplied by 4 at the add.",
               "", "| add after conv | sa(act) | sa(residual) | out sa | ratio | clamp % (wire) |",
               "|---|---|---|---|---|---|"]
        for a in results[labels[0]]["adds"]:
            md.append(f"| {a['after_conv']} | {a['sa_act']:.5g} | {a['sa_slot']:.5g} | "
                      f"{a['out_sa']:.5g} | **{a['ratio']:.2f}×** | {a['clamp'] * 100:.4f} |")
        md += ["", f"`conv_out` (96→3) carries ONE output scale "
               f"`{float(sim.SAOUT[L35['ai']]):.6g}` for three channels whose reference maxima are "
               f"{', '.join('%.4f' % v for v in ch)} — **{max(ch) / (min(ch) or 1):.2f}× spread**, "
               "so the smallest channel runs on the largest channel's LSB. The `coutpc` arm gives "
               "each channel `max/127`.", "",
               "| arm | final PSNR (2.0) | cosine | rel-RMSE | worst layer | `conv_out` cosine |",
               "|---|---|---|---|---|---|"]
        for lb in labels:
            r = results[lb]
            fin = r.get("final_pc") or r.get("final", {})
            w = r.get("worst") or {}
            c35 = next((c for c in r["convs"] if c["idx"] == 35), {})
            md.append(f"| **{lb}** | {fin.get('psnr_db', float('nan')):.2f} dB | "
                      f"{fin.get('cosine', float('nan')):.6f} | "
                      f"{fin.get('rel_rmse', 0) * 100:.2f} % | "
                      f"{w.get('idx')} ({w.get('cosine', float('nan')):.4f}) | "
                      f"{c35.get('cosine', float('nan')):.4f}"
                      + (f" (deq {c35['cosine_deq']:.4f})" if "cosine_deq" in c35 else "") + " |")
        md += ["", "Per-channel dequant is used for every arm's final number above (the `wire` arm "
               "has one scale, so this changes nothing for it); the `conv_out` column is the "
               "layer's own cosine against the reference module output."]
    if oracle_doc:
        orc = oracle_doc
        md += ["", "## P3.17 — inherited vs intrinsic: the same conv on the reference's own input", "",
               "`in` = the walk's own conv input vs the reference module's input at the program's "
               "`SA_in`; `wire` = the conv's output cosine (P3.15/P3.16's column); `oracle` = the "
               "same conv, same weights, same scheme, fed the **reference's** input. A layer whose "
               "`oracle` is ~0.99 is faithful given a good input — its `wire` number is inherited. "
               "One whose `oracle` is ~0.85 loses that much **at the site**.", "",
               "| layer | module | `in` cos | `wire` cos | `oracle` cos | inherited | intrinsic |",
               "|---|---|---|---|---|---|---|"]
        for r in orc["rows"]:
            f = lambda v: "—" if v is None else f"{v:.6f}"
            ft = lambda v: "—" if v is None else f"{v * 100:.2f} pt"
            md.append(f"| {r['idx']} | `{r['name'].split('decoder.')[-1]}` | {f(r['in_cos'])} | "
                      f"{f(r['wire_cos'])} | {f(r['oracle_cos'])} | {ft(r['inherited'])} | "
                      f"{ft(r['intrinsic'])} |")
        md += ["", f"**Mean intrinsic loss {orc['mean_intrinsic'] * 100:.2f} pt** (median "
               f"{orc['median_intrinsic'] * 100:.2f}, max {orc['max_intrinsic'] * 100:.2f}); mean "
               f"inherited {orc['mean_inherited'] * 100:.2f} pt. "
               f"{orc['sites_intrinsic_gt_1pct']}/{orc['n']} sites lose > 1 pt on a perfect input, "
               f"{orc['sites_intrinsic_gt_2pct']}/{orc['n']} > 2 pt. Worst site: "
               + (f"layer {orc['worst_site']['idx']} (`{orc['worst_site']['name']}`) at "
                  f"{orc['worst_site']['oracle_cos']:.6f}." if orc["worst_site"] else "—")]
        l34 = orc.get("layer34")
        if l34:
            md += ["", f"Layer 34 (`{l34['name']}`): input {l34['in_cos']:.6f}, wire output "
                   f"{l34['wire_cos']:.6f}, oracle output {l34['oracle_cos']:.6f} ⇒ intrinsic "
                   f"{l34['intrinsic'] * 100:.2f} pt, inherited {l34['inherited'] * 100:.2f} pt."]
    if glue_doc:
        g = glue_doc
        md += ["", "## P3.18 — the glue, one hop at a time", "",
               "Same convs, same weights, same per-tap scheme in every arm; the only thing that "
               "changes is **which glue ops round to int8**. `gluefp` = every `NORMSILU` / `NORM` / "
               "`SILU` and every residual `ADD` executed in float32 (the state then carries scale "
               "1.0, so the next requant is the conv's own input build and nothing else); "
               "`normfp` and `addfp` remove one class each.", "",
               "| arm | glue in float32 | final PSNR (2.0) | cosine | rel-RMSE | worst conv | int8 clamps |",
               "|---|---|---|---|---|---|---|"]
        for lb in labels:
            r = results[lb]
            fin = r["final"]
            w = r.get("worst") or {}
            md.append(f"| **{lb}** | {'+'.join(r['glue_fp']) or '—'} | {fin['psnr_db']:.2f} dB | "
                      f"{fin['cosine']:.6f} | {fin['rel_rmse'] * 100:.2f} % | "
                      f"{w.get('idx')} ({w.get('cosine', float('nan')):.4f}) | {r['total_clamps']} |")
        md += ["", "`wire` is the validator: it must reproduce P3.17 to the digit "
               "(20.5876 dB / 0.888501 / worst layer 34 at 0.714915).", "",
               "### What each hop costs, by class", "",
               "`hop` = 1 − cos(fp32 result of the op, that same result requantised to the op's "
               "own `out_sa`), priced **at the input the walk handed it**. `INPUT-BUILD` is the "
               "conv's input construction (temporal shuffle / 2× upsample + requant to `SA_in`). "
               "The sums are not additive in dB — they rank the classes and locate the worst site.", "",
               "| class | ops | Σ hop (pt) | mean hop (pt) | max hop (pt) | worst after conv | mean clamp % |",
               "|---|---|---|---|---|---|---|"]
        for k, v in g["by_class"].items():
            md.append(f"| {k} | {v['n']} | {v['sum_pt']:.2f} | {v['mean_pt']:.3f} | "
                      f"{v['max_pt']:.3f} | {v['worst_after_conv']} | "
                      + ("—" if v["mean_clamp_pct"] is None else f"{v['mean_clamp_pct']:.4f}")
                      + " |")
        md += ["", "### Every site, in walk order", "",
               "| op | after conv | module | sa_in | sa_out | requant ratio | hop (pt) | clamp % |",
               "|---|---|---|---|---|---|---|---|"]
        for s in g["sites"]:
            md.append(f"| {s['kind']} | {s['after_conv']} | "
                      f"`{(s['name'] or '').split('decoder.')[-1]}` | {s['sa_in']:.5g} | "
                      + ("—" if s["sa_out"] is None else f"{s['sa_out']:.5g}") + " | "
                      + ("—" if s["g"] is None else f"{s['g']:.3f}") + " | "
                      + ("— (fp32)" if s["hop"] is None else f"{s['hop'] * 100:.3f}") + " | "
                      + ("—" if s["clamp"] is None else f"{s['clamp'] * 100:.4f}") + " |")
    mdp = OUT / f"{args.out}_{ts}.md"
    mdp.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {js}\nwrote {mdp}")
    for lb in labels:
        fin = results[lb].get("final", {})
        fb = results[lb]["first_divergence"]
        print(f"{lb:10s}: " + ("no divergence" if not fb else
              f"first div layer {fb['idx']} ({fb['cosine']:.6f})") +
              ("" if "psnr_db" not in fin else
               f" | final {fin['psnr_db']:.2f} dB / cos {fin['cosine']:.6f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
