"""Exact pinned-lazer difficulty mappings (see runs/development/sim_v1/understand/rules_spec.md section 1).

.osu difficulty values are C# float32 in lazer and widened to double inside
DifficultyRange, so values are first rounded to float32 here.
"""
from __future__ import annotations

import math

import numpy as np

OBJECT_RADIUS = 64.0
MISS_WINDOW = 400.0


def f32(x: float) -> float:
    return float(np.float32(x))


def difficulty_range(d: float, lo: float, mid: float, hi: float) -> float:
    d = f32(d)
    r = (d - 5.0) / 5.0
    if d > 5:
        return mid + (hi - mid) * r
    if d < 5:
        return mid + (mid - lo) * r
    return mid


def scale_from_cs(cs: float) -> float:
    r = (f32(cs) - 5.0) / 5.0
    s = np.float32(1.0 - float(np.float32(0.7)) * r)
    s = np.float32(s / np.float32(2))
    return float(np.float32(s * np.float32(1.00041)))


def radius_from_cs(cs: float) -> float:
    return float(np.float32(OBJECT_RADIUS) * np.float32(scale_from_cs(cs)))


def preempt_from_ar(ar: float) -> float:
    return float(math.trunc(difficulty_range(ar, 1800, 1200, 450)))


def fade_in_from_preempt(preempt: float) -> float:
    return 400.0 * min(1.0, preempt / 450.0)


def hit_windows(od: float) -> tuple[float, float, float]:
    """(great, ok, meh) inclusive half-widths in ms."""
    great = math.floor(difficulty_range(od, 80, 50, 20)) - 0.5
    ok = math.floor(difficulty_range(od, 140, 100, 60)) - 0.5
    meh = math.floor(difficulty_range(od, 200, 150, 100)) - 0.5
    return great, ok, meh
