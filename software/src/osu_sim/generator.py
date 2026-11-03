"""Procedural osu!standard curriculum maps (written as ordinary .osu content).

Every generated map is an OsuFile, so it goes through exactly the same
parse -> beatmap -> compile path as real maps and can be played unchanged by
the real lazer host for parity checks. Difficulty is controlled by a small set
of knobs; ``sample_params(level, rng)`` maps a scalar curriculum level in [0, 1]
to a randomized parameter set.

Drill maps (labelled (session 3), synthesis plan items 1c and 4): ``generate_drill(kind, seed)`` emits a
spinner drill (``spinner_drill_params`` + ``generate``) or a burst drill (``generate_burst_drill``: runs of
1/4 notes, true stacks, kick sliders, slider -> overlapped circle). Same OsuFile path as every other map.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .beatmap import build_path, convert_path
from .difficulty import radius_from_cs, scale_from_cs
from .osufile import OsuFile, RawHitObject, TimingPoint


@dataclass
class GenParams:
    duration_ms: float = 20000.0
    lead_in_ms: tuple[float, float] = (800.0, 2500.0)   # first hit time is preempt + U(lead_in)
    bpm: float = 120.0
    divisors: tuple[tuple[int, float], ...] = ((1, 1.0),)  # (beat divisor, weight): gap = beat/divisor
    rest_prob: float = 0.0          # chance to insert an extra full-beat rest
    cs: float = 4.0
    ar: float = 8.0
    od: float = 6.0
    hp: float = 5.0
    spacing: tuple[float, float] = (60.0, 200.0)   # osu px between consecutive objects (per gap of one beat)
    spacing_per_ms: bool = True     # scale spacing by gap/beat (distance snapping)
    max_jump: float = 350.0
    slider_prob: float = 0.0
    slider_len: tuple[float, float] = (60.0, 200.0)
    slider_repeat_prob: float = 0.15
    slider_curve_probs: tuple[float, float, float] = (0.4, 0.4, 0.2)   # L, P, B
    # labelled (session 3): redraw sliders whose built path leaves the playfield (off: rng stream unchanged)
    slider_in_bounds: bool = False
    sv: tuple[float, float] = (0.8, 1.4)
    slider_multiplier: float = 1.4
    tick_rate: float = 1.0
    spinner_prob: float = 0.0
    spinner_beats: tuple[float, float] = (2.0, 6.0)
    # labelled (session 3): spinner length in ms (snapped to 1/4 beats inside the range); None = spinner_beats
    spinner_ms: tuple[float, float] | None = None
    new_combo_every: tuple[int, int] = (4, 8)
    stream_prob: float = 0.0        # chance a run of 1/4 notes is a tight stream
    stream_len: tuple[int, int] = (4, 12)
    stream_spacing: tuple[float, float] = (15.0, 45.0)
    margin: float = 16.0


def _clip_pos(x, y, m):
    return float(np.clip(x, m, 512 - m)), float(np.clip(y, m, 384 - m))


def _walk(rng, pos, dist, max_jump, m):
    """Random direction step of length dist, reflected/clamped into the playfield."""
    dist = min(dist, max_jump)
    for _ in range(12):
        ang = rng.uniform(0, 2 * math.pi)
        x, y = pos[0] + dist * math.cos(ang), pos[1] + dist * math.sin(ang)
        if m <= x <= 512 - m and m <= y <= 384 - m:
            return x, y
    return _clip_pos(pos[0] + dist * math.cos(ang), pos[1] + dist * math.sin(ang), m)


def _slider_points(rng, start, length, kind, m):
    """Control points (excluding head) for a slider of approximately ``length`` that stays in bounds."""
    for _ in range(20):
        ang = rng.uniform(0, 2 * math.pi)
        if kind == "L":
            end = (start[0] + length * math.cos(ang), start[1] + length * math.sin(ang))
            pts = [end]
        elif kind == "P":
            # arc through a mid point offset perpendicular to the chord
            chord = length * rng.uniform(0.6, 0.95)
            end = (start[0] + chord * math.cos(ang), start[1] + chord * math.sin(ang))
            mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            off = rng.choice([-1, 1]) * chord * rng.uniform(0.15, 0.45)
            mid = (mid[0] - off * math.sin(ang), mid[1] + off * math.cos(ang))
            pts = [mid, end]
        else:  # B: one or two random control points
            n = rng.integers(1, 3)
            pts, cur, a = [], start, ang
            seg = length / (n + 1) * 1.15
            for _ in range(n + 1):
                a += rng.uniform(-1.0, 1.0)
                cur = (cur[0] + seg * math.cos(a), cur[1] + seg * math.sin(a))
                pts.append(cur)
        if all(m <= x <= 512 - m and m <= y <= 384 - m for x, y in pts):
            return [(round(x), round(y)) for x, y in pts]
    end = _clip_pos(start[0] + length * math.cos(ang), start[1] + length * math.sin(ang), m)
    return [(round(end[0]), round(end[1]))]


def generate(rng: np.random.Generator, p: GenParams) -> OsuFile:
    beat = 60000.0 / p.bpm
    preempt = _preempt(p.ar)
    t = preempt + rng.uniform(*p.lead_in_ms)
    end = t + p.duration_ms
    sv = float(rng.uniform(*p.sv))
    timing = [TimingPoint(0.0, beat, 4, 1, 0, 60, True, 0),
              TimingPoint(0.0, -100.0 / sv, 4, 1, 0, 60, False, 0)]
    objs: list[RawHitObject] = []
    pos = (rng.uniform(100, 412), rng.uniform(80, 304))
    combo_left = 0
    div_vals = np.array([d for d, _ in p.divisors])
    div_w = np.array([w for _, w in p.divisors], dtype=float)
    div_w /= div_w.sum()
    px_per_beat = 100 * p.slider_multiplier * sv
    while t < end:
        new_combo = combo_left <= 0
        if new_combo:
            combo_left = int(rng.integers(p.new_combo_every[0], p.new_combo_every[1] + 1))
        typ_nc = 4 if new_combo or not objs else 0
        r = rng.uniform()
        if r < p.spinner_prob and objs:
            if p.spinner_ms is None:
                dur = beat * rng.uniform(*p.spinner_beats)
            else:   # labelled (session 3): one draw, as above, so the rng stream keeps its shape
                q = beat / 4
                n_q = round(rng.uniform(*p.spinner_ms) / q)
                dur = q * min(max(n_q, math.ceil(p.spinner_ms[0] / q)), math.floor(p.spinner_ms[1] / q))
            objs.append(RawHitObject(256, 192, round(t), 8 | 4, 0, end_time=round(t + dur)))
            t += dur + beat * rng.choice([1, 2])
            combo_left = 0
            pos = (256.0, 192.0)
            continue
        if r < p.spinner_prob + p.stream_prob:
            n = int(rng.integers(p.stream_len[0], p.stream_len[1] + 1))
            gap = beat / 4
            sp = rng.uniform(*p.stream_spacing)
            ang = rng.uniform(0, 2 * math.pi)
            curve = rng.uniform(-0.4, 0.4)
            for k in range(n):
                if t >= end:
                    break
                objs.append(RawHitObject(round(pos[0]), round(pos[1]), round(t), 1 | (typ_nc if k == 0 else 0), 0))
                ang += curve
                nx, ny = pos[0] + sp * math.cos(ang), pos[1] + sp * math.sin(ang)
                if not (p.margin <= nx <= 512 - p.margin and p.margin <= ny <= 384 - p.margin):
                    ang += math.pi * rng.uniform(0.5, 1.0)
                    nx, ny = _clip_pos(pos[0] + sp * math.cos(ang), pos[1] + sp * math.sin(ang), p.margin)
                pos = (nx, ny)
                t += gap
            t += beat
            combo_left = 0
            continue
        div = int(rng.choice(div_vals, p=div_w))
        gap = beat / div
        if rng.uniform() < p.rest_prob:
            gap += beat
        if r < p.spinner_prob + p.stream_prob + p.slider_prob:
            kind = "LPB"[int(rng.choice(3, p=np.array(p.slider_curve_probs) / sum(p.slider_curve_probs)))]
            # length snapped to a whole number of 1/2 beats
            want = rng.uniform(*p.slider_len)
            halfs = max(1, round(want / (px_per_beat / 2)))
            length = halfs * px_per_beat / 2
            pts = _slider_points(rng, pos, length, kind, p.margin)
            if p.slider_in_bounds:      # labelled (session 3); off by default: no extra rng draws
                pts = _fit_slider(rng, pos, length, kind, p.margin, pts)
            slides = 2 if rng.uniform() < p.slider_repeat_prob else 1
            if kind == "P" and len(pts) != 2:
                kind = "B"
            objs.append(RawHitObject(round(pos[0]), round(pos[1]), round(t), 2 | typ_nc, 0, curve_type=kind,
                                     curve_points=pts, slides=slides, length=float(length)))
            span = length / px_per_beat * beat
            t += span * slides
            pos = (float(pts[-1][0]), float(pts[-1][1])) if slides % 2 == 1 else pos
            t += gap
        else:
            objs.append(RawHitObject(round(pos[0]), round(pos[1]), round(t), 1 | typ_nc, 0))
            t += gap
        combo_left -= 1
        dist = rng.uniform(*p.spacing)
        if p.spacing_per_ms:
            dist *= min(2.0, gap / beat)
        pos = _walk(rng, pos, dist, p.max_jump, p.margin)
    f = OsuFile()
    f.general.update({"StackLeniency": "0.7", "Mode": "0"})
    f.difficulty.update({"HPDrainRate": p.hp, "CircleSize": p.cs, "OverallDifficulty": p.od, "ApproachRate": p.ar,
                         "SliderMultiplier": p.slider_multiplier, "SliderTickRate": p.tick_rate})
    f.timing_points = timing
    f.hit_objects = objs
    return f


def _preempt(ar: float) -> float:
    return 1200 - 750 * (ar - 5) / 5 if ar > 5 else 1200 - 600 * (ar - 5) / 5


def sample_params(level: float, rng: np.random.Generator) -> GenParams:
    """Curriculum level 0..1 -> randomized generator parameters (monotone in difficulty)."""
    L = float(np.clip(level, 0.0, 1.0))
    lerp = lambda a, b: a + (b - a) * L
    jitter = lambda lo, hi: float(rng.uniform(lo, hi))
    bpm = jitter(lerp(70, 140), lerp(110, 230))
    ar = float(np.clip(jitter(lerp(5.0, 8.5), lerp(7.0, 9.8)), 0, 10))
    od = float(np.clip(jitter(lerp(2.0, 7.0), lerp(5.0, 9.5)), 0, 10))
    cs = float(np.clip(jitter(lerp(2.5, 3.5), lerp(4.0, 5.0)), 0, 10))
    divisors = ((1, 1.0), (2, lerp(0.0, 1.5)), (4, lerp(0.0, 0.4)))
    return GenParams(
        duration_ms=jitter(12000, 25000), bpm=bpm, divisors=divisors, rest_prob=lerp(0.4, 0.1),
        cs=round(cs, 1), ar=round(ar, 1), od=round(od, 1), hp=5.0,
        spacing=(lerp(40, 80), lerp(140, 280)), max_jump=lerp(200, 380),
        slider_prob=lerp(0.0, 0.35) if L > 0.15 else 0.0, slider_len=(60, lerp(120, 260)),
        slider_repeat_prob=lerp(0.0, 0.25), sv=(lerp(0.6, 1.0), lerp(1.0, 2.0)),
        spinner_prob=0.03 if L > 0.3 else 0.0, stream_prob=lerp(0.0, 0.12) if L > 0.4 else 0.0,
        stream_spacing=(lerp(30, 12), lerp(55, 40)))


# ------------------------------------------------------------------------------------------------ drill maps
# labelled (session 3): synthesis plan items 1c (spinner drills) and 4 (burst / stack / short-slider drills), a
# spec 11 C5 task-distribution change. scripts/sim_build_drill_pool.py gives drills pseudo-stars in bands with no
# real maps (spinner [20, 20.5), burst [21, 21.5)), so a StarMixture range over such a band draws only drills.

DRILL_KINDS = ("spinner", "burst")
BURST_RUN_K = ((2, 2, 0.30), (3, 3, 0.25), (4, 5, 0.20), (6, 9, 0.15), (10, 16, 0.10))       # notes per run (lo, hi, p)
BURST_SPACING = ((0.0, 0.0, 0.30), (0.1, 0.5, 0.25), (0.5, 1.0, 0.25), (1.0, 2.0, 0.20))    # radii between run notes


def spinner_drill_params(rng: np.random.Generator) -> GenParams:
    """Spinner drill (plan item 1c): 15-30 s of circles/sliders with spinner_prob 0.3-0.5, spinners of 0.5-4 s,
    OD 7-10, AR 9-10, CS 3.5-5, BPM 150-220. Played by the unchanged ``generate``."""
    duration = float(rng.uniform(15000, 30000))
    bpm = float(rng.uniform(150, 220))
    cs = round(float(rng.uniform(3.5, 5.0)), 1)
    ar = round(float(rng.uniform(9.0, 10.0)), 1)
    od = round(float(rng.uniform(7.0, 10.0)), 1)
    spinner_prob = float(rng.uniform(0.3, 0.5))
    return GenParams(duration_ms=duration, bpm=bpm, divisors=((1, 1.0), (2, 1.0)), rest_prob=0.1, cs=cs, ar=ar, od=od,
                     spacing=(60.0, 200.0), max_jump=300.0, slider_prob=0.25, slider_len=(60.0, 200.0),
                     slider_repeat_prob=0.15, sv=(0.8, 1.4), spinner_prob=spinner_prob, spinner_ms=(500.0, 4000.0),
                     slider_in_bounds=True)


@dataclass
class BurstParams:
    duration_ms: tuple[float, float] = (15000.0, 30000.0)
    lead_in_ms: tuple[float, float] = (800.0, 2500.0)
    bpm: tuple[float, float] = (160.0, 240.0)          # 1/4 notes 62.5-94 ms apart
    cs: tuple[float, float] = (3.5, 5.0)
    ar: tuple[float, float] = (9.0, 10.0)
    od: tuple[float, float] = (8.0, 10.0)
    hp: float = 5.0
    run_k: tuple = BURST_RUN_K
    spacing: tuple = BURST_SPACING
    curve: float = 0.4                                  # max direction change (rad) per run note
    jump: tuple[float, float] = (1.5, 5.0)             # radii from a run's last object to the next run's first
    rest_beats: tuple[float, ...] = (0.5, 1.0)          # gap after a run (uniform choice)
    kick_prob: float = 0.15         # run -> 1/4-beat kick slider (< 100 ms at >= 160 BPM) followed by k-1 circles
    overlap_prob: float = 0.10      # run -> slider then one circle within 1 radius of its end, 1/4 beat later
    overlap_quarters: tuple[int, int] = (2, 8)          # that slider's length in 1/4 beats
    max_slider_len: float = 250.0
    slider_multiplier: tuple[float, ...] = (1.4, 1.6, 1.8)
    sv: tuple[float, float] = (0.8, 1.4)
    new_combo_prob: float = 0.5     # chance a run starts a new combo (always after 8+ objects)
    margin: float = 16.0


def _pick(rng, table):
    """(lo, hi, weight) rows: pick a row by weight, then uniform inside it (inclusive integers for int rows)."""
    w = np.array([r[2] for r in table], dtype=float)
    lo, hi, _ = table[int(rng.choice(len(table), p=w / w.sum()))]
    if isinstance(lo, int) and isinstance(hi, int):
        return int(rng.integers(lo, hi + 1))
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def _stream_step(rng, pos, ang, curve, dist, m):
    """Next stream note ``dist`` px along a gently curving direction, turned back at the playfield edge."""
    ang += curve
    nx, ny = pos[0] + dist * math.cos(ang), pos[1] + dist * math.sin(ang)
    if not (m <= nx <= 512 - m and m <= ny <= 384 - m):
        ang += math.pi * rng.uniform(0.5, 1.0)
        nx, ny = _clip_pos(pos[0] + dist * math.cos(ang), pos[1] + dist * math.sin(ang), m)
    return (nx, ny), ang


def _slider_end(head, kind, pts, length, version=OsuFile.version):
    """Unstacked end of a one-span slider, computed exactly as build_beatmap does (osu px)."""
    hx, hy = float(int(head[0])), float(int(head[1]))
    e = build_path(convert_path(kind, pts, (hx, hy), version), length).position_at(1.0)
    return float(np.float32(hx) + e[0]), float(np.float32(hy) + e[1])


def _path_in_bounds(head, kind, pts, length, version=OsuFile.version):
    """Does the slider path, built as build_beatmap builds it (head rounded like RawHitObject), stay in 512x384?"""
    hx, hy = float(round(head[0])), float(round(head[1]))
    v = build_path(convert_path(kind, pts, (hx, hy), version), length).vertices
    return bool(((v[:, 0] + hx >= 0) & (v[:, 0] + hx <= 512) & (v[:, 1] + hy >= 0) & (v[:, 1] + hy <= 384)).all())


def _fit_slider(rng, start, length, kind, m, pts, tries=10):
    """Redraw ``pts`` with _slider_points (up to ``tries`` times) until the built path stays on the playfield:
    _slider_points only bounds the control points, so a P arc can bulge out (about 1 slider in 4000)."""
    for _ in range(tries):
        if _path_in_bounds(start, "B" if kind == "P" and len(pts) != 2 else kind, pts, length):
            break
        pts = _slider_points(rng, start, length, kind, m)
    return pts


def generate_burst_drill(rng: np.random.Generator, p: BurstParams | None = None) -> OsuFile:
    """Burst / stack / short-slider drill (plan item 4): runs of k 1/4 notes spaced in circle radii (0 = true stack:
    identical x,y, offset by lazer's stacking like a real stack), each followed by a ``jump``-radii jump and a 1/2-
    or 1-beat rest; kick-slider runs and slider -> overlapped-circle runs mixed in."""
    p = p or BurstParams()
    f = OsuFile()
    bpm = float(rng.uniform(*p.bpm))
    beat = 60000.0 / bpm
    q = beat / 4
    cs = round(float(rng.uniform(*p.cs)), 1)
    ar = round(float(rng.uniform(*p.ar)), 1)
    od = round(float(rng.uniform(*p.od)), 1)
    sm = float(rng.choice(p.slider_multiplier))
    sv = float(rng.uniform(*p.sv))
    rad = radius_from_cs(cs)
    stack_px = 6.4 * scale_from_cs(cs)      # lazer stack offset per stack level (x and y)
    px_quarter = 100 * sm * sv / 4          # slider px per 1/4 beat
    t = _preempt(ar) + rng.uniform(*p.lead_in_ms)
    end = t + rng.uniform(*p.duration_ms)
    m = p.margin
    objs: list[RawHitObject] = []
    pos = (rng.uniform(100, 412), rng.uniform(80, 304))
    since_nc = 0

    def circle(at, time, nc):
        objs.append(RawHitObject(round(at[0]), round(at[1]), round(time), 1 | (4 if nc else 0), 0))

    def slider(head, time, nc, kind, pts, length):
        objs.append(RawHitObject(head[0], head[1], round(time), 2 | (4 if nc else 0), 0, curve_type=kind,
                                 curve_points=pts, slides=1, length=float(length)))

    while t < end:
        u = float(rng.uniform())
        mode = "kick" if u < p.kick_prob else ("overlap" if u < p.kick_prob + p.overlap_prob else "run")
        k = _pick(rng, p.run_k)
        sp = _pick(rng, p.spacing) * rad
        nc = (rng.uniform() < p.new_combo_prob) or not objs or since_nc >= 8
        # a true stack of n objects piles up to n * stack_px away from its base: keep it on the playfield
        pad = m + (k + 1) * stack_px if sp == 0 else m
        if objs:
            d = float(rng.uniform(*p.jump)) * rad
            for _ in range(4):
                nxt = _walk(rng, pos, d, math.inf, pad)
                if math.hypot(nxt[0] - pos[0], nxt[1] - pos[1]) >= 0.9 * d:
                    break
            pos = nxt
        else:
            pos = _clip_pos(pos[0], pos[1], pad)
        ang = rng.uniform(0, 2 * math.pi)
        curve = rng.uniform(-p.curve, p.curve)
        n0 = len(objs)
        if mode == "run":
            for i in range(k):
                if i:
                    pos, ang = _stream_step(rng, pos, ang, curve, sp, pad)
                    t += q
                circle(pos, t, nc and i == 0)
        elif mode == "kick":
            head = (round(pos[0]), round(pos[1]))
            pts = _fit_slider(rng, head, px_quarter, "L", pad, _slider_points(rng, head, px_quarter, "L", pad))
            slider(head, t, nc, "L", pts, px_quarter)
            pos = _slider_end(head, "L", pts, px_quarter, f.version)
            ang = math.atan2(pos[1] - head[1], pos[0] - head[0])
            t += q                                   # slider end
            for i in range(max(1, k - 1)):
                pos, ang = _stream_step(rng, pos, ang, curve, sp, pad)
                t += q
                circle(pos, t, False)
        else:
            nq = int(rng.integers(p.overlap_quarters[0], p.overlap_quarters[1] + 1))
            nq = max(1, min(nq, int(p.max_slider_len // px_quarter)))
            kind = "L" if rng.uniform() < 0.5 else "P"
            head = (round(pos[0]), round(pos[1]))
            pts = _fit_slider(rng, head, nq * px_quarter, kind, m, _slider_points(rng, head, nq * px_quarter, kind, m))
            if kind == "P" and len(pts) != 2:
                kind = "B"
            slider(head, t, nc, kind, pts, nq * px_quarter)
            ex, ey = _slider_end(head, kind, pts, nq * px_quarter, f.version)
            d = float(rng.uniform(0, 1)) * rad
            for _ in range(8):          # a direction that keeps the circle inside the margin (else clip)
                a = rng.uniform(0, 2 * math.pi)
                c = (ex + d * math.cos(a), ey + d * math.sin(a))
                if m <= c[0] <= 512 - m and m <= c[1] <= 384 - m:
                    break
            pos = _clip_pos(c[0], c[1], m)
            t += nq * q + q                          # 1/4 beat after the slider end
            circle(pos, t, False)
        since_nc = (0 if nc else since_nc) + len(objs) - n0
        t += beat * float(rng.choice(p.rest_beats))
    f.general.update({"StackLeniency": "0.7", "Mode": "0"})
    f.difficulty.update({"HPDrainRate": p.hp, "CircleSize": cs, "OverallDifficulty": od, "ApproachRate": ar,
                         "SliderMultiplier": sm, "SliderTickRate": 1.0})
    f.timing_points = [TimingPoint(0.0, beat, 4, 1, 0, 60, True, 0),
                       TimingPoint(0.0, -100.0 / sv, 4, 1, 0, 60, False, 0)]
    f.hit_objects = objs
    return f


def generate_drill(kind: str, seed: int) -> OsuFile:
    """Drill map of ``kind`` (DRILL_KINDS); deterministic in (kind, seed)."""
    rng = np.random.default_rng(seed)
    if kind == "spinner":
        f = generate(rng, spinner_drill_params(rng))
    elif kind == "burst":
        f = generate_burst_drill(rng)
    else:
        raise ValueError(f"unknown drill kind {kind!r}")
    f.metadata["Version"] = f"drill-{kind}"
    return f
