"""Scripted reference players (diagnostics only; never used to train the policy).

Autopilot: at each decision (applied at t_prev) aims at the object that should be hit next, presses on
the step whose time is closest to the hit time (alternating Z/X), follows slider balls with the key held,
and spins spinners. Used for rules self-tests and real-host parity traces.
"""
from __future__ import annotations

import math

import numpy as np

from .env import CLIENT_SCALE, ORIGIN_X, ORIGIN_Y


def gamefield_to_action(x, y):
    cx = ORIGIN_X + CLIENT_SCALE * x
    cy = ORIGIN_Y + CLIENT_SCALE * y
    return np.clip(np.array(((cx - 160) * 2 / 959 - 1, cy * 2 / 719 - 1), dtype=np.float64), -1, 1)


class Autopilot:
    def __init__(self, pool, dt, *, offset_ms=0.0, miss_prob=0.0, break_prob=0.0, rng=None, spin_deg=120.0):
        self.pool = pool
        self.dt = dt
        self.offset = offset_ms
        self.miss_prob = miss_prob
        self.break_prob = break_prob
        self.rng = rng or np.random.default_rng(0)
        self.spin_deg = spin_deg
        self.last_key = 1
        self.spin_angle = 0.0
        self.skip = set()

    def _curve(self, g, prog):
        o, p = self.pool.objs, self.pool.path
        off, cnt = int(o["path_off"][g]), int(o["path_cnt"][g])
        dist = float(o["distance"][g])
        spans = int(o["span_count"][g])
        q = prog * spans
        span = min(int(math.floor(q)), spans - 1)
        frac = q - span if q < spans else 1.0
        pr = 1 - frac if span % 2 == 1 else frac
        d = min(max(pr, 0.0), 1.0) * dist
        cum = p["cum"][off:off + cnt]
        i = int(np.searchsorted(cum, d, side="left"))
        if i <= 0:
            return float(p["x"][off]), float(p["y"][off])
        if i >= cnt:
            return float(p["x"][off + cnt - 1]), float(p["y"][off + cnt - 1])
        d0, d1 = cum[i - 1], cum[i]
        w = 0.0 if abs(d1 - d0) <= 1e-7 else (d - d0) / (d1 - d0)
        return (float(p["x"][off + i - 1] + (p["x"][off + i] - p["x"][off + i - 1]) * w),
                float(p["y"][off + i - 1] + (p["y"][off + i] - p["y"][off + i - 1]) * w))

    def act(self, t_prev, lo, hi, prev_key):
        """Return (action[2], key) for the decision applied at t_prev, objects global ids [lo, hi)."""
        o = self.pool.objs
        dt = self.dt
        # active slider / spinner at t_prev?
        for g in range(lo, hi):
            s, e, k = float(o["start"][g]), float(o["end"][g]), int(o["kind"][g])
            if k == 1 and s - dt / 2 < t_prev <= e + 5:
                prog = min(max((t_prev + dt * 0.5 - s) / max(e - s, 1e-6), 0.0), 1.0)
                x, y = self._curve(g, prog)
                key = prev_key if prev_key else self.last_key
                if g in self.skip:
                    return gamefield_to_action(x + 200, y), 0
                return gamefield_to_action(x, y), key
            if k == 2 and s - dt <= t_prev < e:
                self.spin_angle += math.radians(self.spin_deg)
                r = 120.0
                return gamefield_to_action(256 + r * math.cos(self.spin_angle), 192 + r * math.sin(self.spin_angle)), \
                    (prev_key if prev_key else self.last_key)
        # next object to hit
        for g in range(lo, hi):
            s, k = float(o["start"][g]), int(o["kind"][g])
            if k == 2 or s + self.offset < t_prev - dt / 2:
                continue
            x, y = float(o["x"][g]), float(o["y"][g])
            if abs(t_prev - (s + self.offset)) <= dt / 2:
                if g not in self.skip and self.rng.random() < self.miss_prob:
                    self.skip.add(g)
                if g in self.skip:
                    return gamefield_to_action(x + 150, y + 150), 0
                self.last_key = 2 if self.last_key == 1 else 1
                if k == 1 and self.rng.random() < self.break_prob:
                    self.skip.add(g)  # will press the head then leave the slider
                    self.skip.discard(g)
                return gamefield_to_action(x, y), self.last_key
            return gamefield_to_action(x, y), 0
        return gamefield_to_action(256, 192), 0
