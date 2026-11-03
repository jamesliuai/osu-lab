"""Adaptive difficulty curriculum over a mixed map pool (spec Section 11: distribution changes, never actions).

Each map has a difficulty d (real: native star rating; generated: pseudo-stars). A frontier F rises when
the agent's recent per-object failure rate on maps in the frontier band [F - band, F] is low. Episodes
are sampled ~75% from the frontier band and ~25% from everything easier (retention, spec: 80/20).
"""
from __future__ import annotations

import numpy as np


class StarCurriculum:
    def __init__(self, stars, *, start=1.0, band=1.0, step_up=0.1, step_down=0.05, max_frontier=7.5,
                 up_fail=0.03, down_fail=0.20, window=400, retain=0.25, min_band_maps=20, generated=None,
                 gen_share_decay=4.0):
        self.d = np.asarray(stars, dtype=np.float64)
        self.F = float(start)
        self.band, self.step_up, self.step_down = band, step_up, step_down
        self.max_frontier = max_frontier
        self.up_fail, self.down_fail = up_fail, down_fail
        self.window, self.retain = window, retain
        self.min_band_maps = min_band_maps
        self.generated = np.zeros(len(self.d), bool) if generated is None else np.asarray(generated, bool)
        self.gen_share_decay = gen_share_decay
        self.hist = []          # (difficulty, objects, failures)
        self.changes = 0

    def _band(self):
        lo = self.F - self.band
        idx = np.nonzero((self.d <= self.F) & (self.d >= lo))[0]
        if len(idx) < self.min_band_maps:
            order = np.argsort(np.abs(self.d - self.F))
            idx = order[:self.min_band_maps]
        return idx

    def sample(self, rng, k):
        band = self._band()
        easier = np.nonzero(self.d < self.F - self.band)[0]
        out = np.empty(k, dtype=np.int64)
        for i in range(k):
            if len(easier) and rng.random() < self.retain:
                out[i] = easier[rng.integers(len(easier))]
            else:
                out[i] = band[rng.integers(len(band))]
        return out

    def report(self, map_ids, rec):
        for m in map_ids:
            fails = rec["breaks"] + rec["tail_miss"]
            self.hist.append((self.d[m], max(rec["objects"], 1), fails))
        if len(self.hist) > 4 * self.window:
            self.hist = self.hist[-2 * self.window:]
        self._adapt()

    def _adapt(self):
        recent = [h for h in self.hist[-self.window:] if h[0] >= self.F - self.band]
        if len(recent) < self.window // 4:
            return
        objs = sum(h[1] for h in recent)
        fails = sum(h[2] for h in recent)
        rate = fails / max(objs, 1)
        if rate < self.up_fail and self.F < self.max_frontier:
            self.F = min(self.max_frontier, self.F + self.step_up)
            self.hist = []
            self.changes += 1
        elif rate > self.down_fail and self.F > 1.0:
            self.F = max(1.0, self.F - self.step_down)
            self.hist = []
            self.changes += 1

    def state(self):
        recent = [h for h in self.hist[-self.window:] if h[0] >= self.F - self.band]
        objs = sum(h[1] for h in recent)
        fails = sum(h[2] for h in recent)
        return dict(frontier=round(self.F, 3), band_fail_rate=round(fails / max(objs, 1), 4), n=len(recent))

    def load(self, state):
        self.F = float(state.get("frontier", self.F))


class StarMixture:
    """Polish-phase sampler (labelled executive decision, spec Section 11 task distribution): fixed shares over
    star ranges, focused on the target difficulty, with PLR-style map priority inside each range.

    ranges: [[lo, hi, share], ...]; within a range a map is drawn uniformly with probability 1 - prioritized,
    else proportionally to its recent per-object failure EMA (+ floor). Unseen maps start at `init_fail`.
    """

    def __init__(self, stars, *, ranges, prioritized=0.5, init_fail=0.1, ema=0.2, floor=0.01, generated=None,
                 window=2000, gen_cap=1.0, **_ignored):
        self.d = np.asarray(stars, dtype=np.float64)
        self.ranges = [(float(lo), float(hi), float(w)) for lo, hi, w in ranges]
        self.idx = [np.nonzero((self.d >= lo) & (self.d < hi))[0] for lo, hi, _ in self.ranges]
        keep = [i for i, ix in enumerate(self.idx) if len(ix)]
        self.ranges = [self.ranges[i] for i in keep]
        self.idx = [self.idx[i] for i in keep]
        w = np.array([r[2] for r in self.ranges])
        self.share = w / w.sum()
        self.prioritized, self.ema, self.floor = prioritized, ema, floor
        self.fail = np.full(len(self.d), float(init_fail))
        self.window = window
        self.hist = []          # (range index, objects, failures)
        # generated maps are drawn with probability min(their share of the range, gen_cap)
        gen = np.zeros(len(self.d), bool) if generated is None else np.asarray(generated, bool)
        self.parts = []
        for ix in self.idx:
            g, r = ix[gen[ix]], ix[~gen[ix]]
            p_gen = 0.0 if not len(g) else (1.0 if not len(r) else min(len(g) / len(ix), gen_cap))
            self.parts.append((g, r, p_gen))

    def sample(self, rng, k):
        out = np.empty(k, dtype=np.int64)
        rs = rng.choice(len(self.ranges), size=k, p=self.share)
        for i, r in enumerate(rs):
            g, real, p_gen = self.parts[r]
            ix = g if rng.random() < p_gen else real
            if rng.random() < self.prioritized:
                p = self.fail[ix] + self.floor
                out[i] = ix[rng.choice(len(ix), p=p / p.sum())]
            else:
                out[i] = ix[rng.integers(len(ix))]
        return out

    def _range_of(self, m):
        for i, (lo, hi, _) in enumerate(self.ranges):
            if lo <= self.d[m] < hi:
                return i
        return -1

    def report(self, map_ids, rec):
        fails = rec["breaks"] + rec["tail_miss"]
        objs = max(rec["objects"], 1)
        for m in map_ids:
            self.fail[m] = (1 - self.ema) * self.fail[m] + self.ema * min(fails / objs, 1.0)
            self.hist.append((self._range_of(m), objs, fails))
        if len(self.hist) > 2 * self.window:
            self.hist = self.hist[-self.window:]

    def state(self):
        st = {"frontier": None}
        recent = self.hist[-self.window:]
        for i, (lo, hi, _) in enumerate(self.ranges):
            h = [x for x in recent if x[0] == i]
            o = sum(x[1] for x in h)
            st[f"fail_{lo:g}-{hi:g}"] = round(sum(x[2] for x in h) / max(o, 1), 4)
        return st

    def full_state(self):
        return {"map_fail": self.fail.copy()}

    def load(self, state):
        if "map_fail" in state and len(state["map_fail"]) == len(self.fail):
            self.fail = np.asarray(state["map_fail"], dtype=np.float64)
        elif "map_fail" in state and len(state["map_fail"]) < len(self.fail):
            # labelled (session 3): a pool that appends maps to the old one (train_mix3 = train_mix2 + drills) keeps
            # the old maps as a prefix; restore their failure EMA, the new maps keep init_fail
            mf = np.asarray(state["map_fail"], dtype=np.float64)
            self.fail[:len(mf)] = mf
            print(f"[curriculum] map_fail: restored {len(mf)} saved maps as a prefix of {len(self.fail)} "
                  f"(pool grew; {len(self.fail) - len(mf)} new maps keep init_fail)", flush=True)
