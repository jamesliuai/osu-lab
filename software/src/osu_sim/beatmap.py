"""OsuFile -> lazer-equivalent osu!standard objects (pinned lazer aaa75b7c / framework 247dee98).

Ports (see runs/development/sim_v1/understand/rules_spec.md sections 1, 4, 5, 6, 9):
  * LegacyBeatmapDecoder timing/difficulty rules, ConvertHitObjectParser path segments,
  * framework PathApproximator (B-spline/Bezier, perfect arc, catmull) + SliderPath length fit,
  * Slider velocity/tick distance, SliderEventGenerator nested events,
  * OsuBeatmapProcessor combo indices and stacking (v>=6 and old).
Float32 arithmetic is used where lazer uses C# float (positions, path vertices).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .difficulty import (MISS_WINDOW, difficulty_range, fade_in_from_preempt, hit_windows, preempt_from_ar,
                         radius_from_cs, scale_from_cs, f32)
from .osufile import OsuFile

F = np.float32
CIRCLE, SLIDER, SPINNER = 0, 1, 2
EV_TICK, EV_REPEAT, EV_TAIL = 0, 1, 2
TAIL_LENIENCY = -36.0


# ----------------------------------------------------------------------------------------- path math
def _v(x, y):
    return np.array((x, y), dtype=F)


def _bezier_flat(p):
    lim = F(0.25) * F(0.25) * F(4)
    for i in range(1, len(p) - 1):
        d = p[i - 1] - F(2) * p[i] + p[i + 1]
        if F(d[0] * d[0] + d[1] * d[1]) > lim:
            return False
    return True


def _bezier_subdivide(cp, count):
    mid = cp.copy()
    l = np.empty_like(cp)
    r = np.empty_like(cp)
    for i in range(count):
        l[i] = mid[0]
        r[count - i - 1] = mid[count - i - 1]
        for j in range(count - i - 1):
            mid[j] = (mid[j] + mid[j + 1]) / F(2)
    return l, r


def _bezier_approximate(cp, out, count):
    l, r = _bezier_subdivide(cp, count)
    big = np.empty((2 * count - 1, 2), dtype=F)
    big[:count] = l
    for i in range(count - 1):
        big[count + i] = r[i + 1]
    out.append(cp[0].copy())
    for i in range(1, count - 1):
        idx = 2 * i
        out.append(F(0.25) * (big[idx - 1] + F(2) * big[idx] + big[idx + 1]))


def _bspline_to_bezier(points, degree):
    pts = points.copy()
    n = len(pts) - 1
    if degree == n:
        return [pts]
    segs = []
    for i in range(n - degree):
        sub = np.empty((degree + 1, 2), dtype=F)
        sub[0] = pts[i]
        for j in range(degree - 1):
            sub[j + 1] = pts[i + 1]
            for k in range(1, degree - j):
                l = min(k, n - degree - i)
                pts[i + k] = (F(l) * pts[i + k] + pts[i + k + 1]) / F(l + 1)
        sub[degree] = pts[i + 1]
        segs.append(sub)
    segs.append(pts[n - degree:].copy())
    return segs


def bspline_to_piecewise_linear(points, degree):
    points = np.asarray(points, dtype=F)
    if len(points) < 2:
        return [p.copy() for p in points]
    degree = min(degree, len(points) - 1)
    out = []
    # stack order: Stack<> built by pushing in order then reversed -> first segment processed first
    stack = list(reversed(_bspline_to_bezier(points, degree)))
    while stack:
        parent = stack.pop()
        if _bezier_flat(parent):
            _bezier_approximate(parent, out, degree + 1)
            continue
        l, r = _bezier_subdivide(parent, degree + 1)
        stack.append(r)
        stack.append(l)
    out.append(points[-1].copy())
    return out


def _catmull_point(v1, v2, v3, v4, t):
    t = F(t)
    t2 = t * t
    t3 = t * t2
    h = F(0.5)
    return h * (F(2) * v2 + (-v1 + v3) * t + (F(2) * v1 - F(5) * v2 + F(4) * v3 - v4) * t2
                + (-v1 + F(3) * v2 - F(3) * v3 + v4) * t3)


def catmull_to_piecewise_linear(points):
    points = np.asarray(points, dtype=F)
    res = []
    n = len(points)
    for i in range(n - 1):
        v1 = points[i - 1] if i > 0 else points[i]
        v2 = points[i]
        v3 = points[i + 1] if i < n - 1 else v2 + v2 - v1
        v4 = points[i + 2] if i < n - 2 else v3 + v3 - v2
        for c in range(50):
            res.append(_catmull_point(v1, v2, v3, v4, F(c) / F(50)))
            res.append(_catmull_point(v1, v2, v3, v4, F(c + 1) / F(50)))
    return res


def _almost_zero_f(x):
    return abs(float(x)) <= 1e-3


def circular_arc(points):
    """Returns (valid, thetaStart, thetaRange, direction, radius(f32), centre(f32 vec))."""
    a, b, c = (np.asarray(p, dtype=F) for p in points[:3])
    cross = F((b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1]))
    if _almost_zero_f(cross):
        return (False,) + (0,) * 5
    d = F(2) * F(a[0] * (b - c)[1] + b[0] * (c - a)[1] + c[0] * (a - b)[1])
    aSq, bSq, cSq = F(a @ a), F(b @ b), F(c @ c)
    cx = F(aSq * (b - c)[1] + bSq * (c - a)[1] + cSq * (a - b)[1])
    cy = F(aSq * (c - b)[0] + bSq * (a - c)[0] + cSq * (b - a)[0])
    centre = np.array((cx, cy), dtype=F) / d
    dA = a - centre
    dC = c - centre
    radius = F(math.sqrt(float(F(dA @ dA))))
    t0 = math.atan2(float(dA[1]), float(dA[0]))
    t1 = math.atan2(float(dC[1]), float(dC[0]))
    while t1 < t0:
        t1 += 2 * math.pi
    direction = 1.0
    rng = t1 - t0
    ortho = c - a
    ortho = np.array((ortho[1], -ortho[0]), dtype=F)
    if float(F(ortho @ (b - a))) < 0:
        direction = -1.0
        rng = 2 * math.pi - rng
    return True, t0, rng, direction, radius, centre


def arc_to_piecewise_linear(points):
    valid, t0, rng, direction, radius, centre = circular_arc(points)
    if not valid:
        return bspline_to_piecewise_linear(points, max(1, len(points) - 1))
    if F(2) * radius <= F(0.1):
        n = 2
    else:
        n = max(2, int(math.ceil(rng / (2 * math.acos(float(F(1) - F(0.1) / radius))))))
    out = []
    for i in range(n):
        fract = i / (n - 1)
        theta = t0 + direction * fract * rng
        o = np.array((F(math.cos(theta)), F(math.sin(theta))), dtype=F) * radius
        out.append(centre + o)
    return out


@dataclass
class SliderPath:
    vertices: np.ndarray        # [V,2] float32, relative to head
    cumulative: np.ndarray      # [V] float64
    distance: float

    def position_at(self, progress: float) -> np.ndarray:
        d = min(max(progress, 0.0), 1.0) * self.distance
        cum = self.cumulative
        if len(self.vertices) == 0:
            return _v(0, 0)
        i = int(np.searchsorted(cum, d, side="left"))
        if i <= 0:
            return self.vertices[0].copy()
        if i >= len(self.vertices):
            return self.vertices[-1].copy()
        d0, d1 = cum[i - 1], cum[i]
        if abs(d1 - d0) <= 1e-7:
            return self.vertices[i - 1].copy()
        w = (d - d0) / (d1 - d0)
        p0, p1 = self.vertices[i - 1], self.vertices[i]
        return p0 + (p1 - p0) * F(w)


def _sub_path(points, kind, degree=None):
    """kind: 'L','P','C','B'."""
    if kind == "L":
        return [np.asarray(p, dtype=F).copy() for p in points], 0.0
    if kind == "P" and len(points) == 3:
        valid, t0, rng, direction, radius, centre = circular_arc(points)
        if valid:
            sub = 2 if F(2) * radius <= F(0.1) else max(2, int(math.ceil(rng / (2.0 * math.acos(float(F(1) - F(0.1) / radius))))))
            if sub < 1000:
                res = arc_to_piecewise_linear(points)
                if res:
                    return res, 0.0
    if kind == "C":
        sub = catmull_to_piecewise_linear(points)
        opt, last_start, removed, opt_len = [], None, 0.0, 0.0
        for i, p in enumerate(sub):
            if last_start is None:
                opt.append(p)
                last_start = p
                continue
            dist = float(np.float32(np.linalg.norm((last_start - p).astype(np.float32))))
            removed += float(np.float32(np.linalg.norm((sub[i - 1] - p).astype(np.float32))))
            if dist > 6 or (i + 1) % 100 == 0 or i == len(sub) - 1:
                opt.append(p)
                opt_len += removed - dist
                last_start = None
                removed = 0.0
        return opt, opt_len
    return bspline_to_piecewise_linear(points, degree if degree is not None else len(points)), 0.0


def build_path(control, expected_distance):
    """control: list of (pos f32[2], type or None). Returns SliderPath (SliderPath.calculatePath/Length)."""
    verts = [c[0] for c in control]
    path: list[np.ndarray] = []
    opt_len = 0.0
    start = 0
    for i in range(len(control)):
        if control[i][1] is None and i < len(control) - 1:
            continue
        seg = verts[start:i + 1]
        kind = control[start][1] or "L"
        if len(seg) == 1:
            path.append(seg[0].copy())
        elif len(seg) > 1:
            sub, ol = _sub_path(seg, kind)
            opt_len += ol
            skip = bool(path) and bool(sub) and np.array_equal(path[-1], sub[0])
            path.extend(sub[1:] if skip else sub)
        start = i
    # calculateLength
    length = opt_len
    cum = [0.0]
    for i in range(len(path) - 1):
        diff = (path[i + 1] - path[i]).astype(F)
        length += float(F(math.sqrt(float(F(diff[0] * diff[0] + diff[1] * diff[1])))))
        cum.append(length)
    if expected_distance is not None and length != expected_distance:
        if len(path) >= 2 and np.array_equal(path[-1], path[-2]) and expected_distance > length:
            cum.append(length)
            cum = cum[:len(path)] if len(cum) > len(path) else cum
            return SliderPath(np.array(path, dtype=F), np.array(cum[:len(path)]), length)
        cum.pop()
        end = len(path) - 1
        if length > expected_distance:
            while cum and cum[-1] >= expected_distance:
                cum.pop()
                path.pop(end)
                end -= 1
        if end <= 0:
            cum.append(0.0)
            return SliderPath(np.array(path, dtype=F).reshape(-1, 2), np.array(cum), 0.0)
        d = (path[end] - path[end - 1]).astype(F)
        nrm = F(math.sqrt(float(F(d[0] * d[0] + d[1] * d[1]))))
        dirv = d / nrm if nrm > 0 else d
        path[end] = path[end - 1] + dirv * F(expected_distance - cum[-1])
        cum.append(expected_distance)
        length = expected_distance
    return SliderPath(np.array(path, dtype=F).reshape(-1, 2), np.array(cum, dtype=np.float64), cum[-1] if cum else 0.0)


def convert_path(curve_type, curve_points, head, version):
    """ConvertHitObjectParser.convertPathString for a v<128 legacy slider. Returns control list."""
    hx, hy = head
    pts = [_v(0, 0)] + [_v(float(int(x)) - hx, float(int(y)) - hy) for x, y in curve_points]
    kind = curve_type if curve_type in ("B", "L", "P", "C") else "C"
    # convertPoints with endPoint=None (single explicit segment in legacy files)
    if kind == "P":
        if len(pts) != 3:
            kind = "B"
        else:
            p0, p1, p2 = pts
            if _almost_zero_f(F((p1[1] - p0[1]) * (p2[0] - p0[0]) - (p1[0] - p0[0]) * (p2[1] - p0[1]))):
                kind = "L"
    types = [None] * len(pts)
    types[0] = kind
    segs = []
    start, end = 0, 0
    while True:
        end += 1
        if end >= len(pts):
            break
        if not np.array_equal(pts[end], pts[end - 1]):
            continue
        if kind == "C" and end > 1:
            continue
        if end == len(pts) - 1:
            continue
        types[end - 1] = kind
        segs.append(list(range(start, end)))
        start = end + 1
    if start < end:
        segs.append(list(range(start, end)))
    control = []
    for seg in segs:
        for i in seg:
            control.append((pts[i], types[i]))
    return control


# ------------------------------------------------------------------------------------- timing
class Timing:
    def __init__(self, osu: OsuFile):
        offset = 24.0 if osu.version < 5 else 0.0
        red, green = [], []   # (time, value) ; green holds (time, sv, generate_ticks)
        pending, pending_time = [], None

        def flush():
            # for each type the LAST pending point of that type wins (timing ones inserted at front)
            tim = [p for p in pending if p[0] == "t"]
            dif = [p for p in pending if p[0] == "d"]
            if tim:
                red.append((pending_time, tim[-1][1]))  # inserted at front; walking backwards keeps the FIRST red line
            if dif:
                green.append((pending_time, dif[-1][1], dif[-1][2]))

        for tp in osu.timing_points:
            t = tp.time + offset
            if pending_time is not None and t != pending_time:
                flush()
                pending = []
            pending_time = t
            beat = tp.beat_length
            sv = 100.0 / -beat if beat < 0 else 1.0
            sv = min(max(sv, 0.1), 10.0)
            gen = not math.isnan(beat)
            if tp.uninherited:
                if math.isnan(beat):
                    raise ValueError("NaN beat length on a timing point")
                pending.insert(0, ("t", min(max(beat, 6.0), 60000.0)))
            pending.append(("d", sv, gen))
        if pending:
            flush()
        # remove redundant difficulty points like lazer? (lookup equivalent without it)
        self.red_t = np.array([r[0] for r in red]) if red else np.array([0.0])
        self.red_v = np.array([r[1] for r in red]) if red else np.array([1000.0])
        self.green_t = np.array([g[0] for g in green])
        self.green_sv = np.array([g[1] for g in green])
        self.green_gen = np.array([g[2] for g in green], dtype=bool)

    def beat_length_at(self, t):
        i = int(np.searchsorted(self.red_t, t, side="right")) - 1
        return float(self.red_v[max(i, 0)])

    def difficulty_at(self, t):
        if len(self.green_t) == 0:
            return 1.0, True
        i = int(np.searchsorted(self.green_t, t, side="right")) - 1
        if i < 0:
            return 1.0, True
        return float(self.green_sv[i]), bool(self.green_gen[i])


# ------------------------------------------------------------------------------------- objects
@dataclass
class HitObj:
    kind: int
    start: float
    x: float                     # unstacked position (float32 values)
    y: float
    new_combo: bool = False
    combo_offset: int = 0
    end: float = 0.0
    # slider
    path: SliderPath | None = None
    span_count: int = 1
    velocity: float = 0.0
    tick_distance: float = 0.0
    span_duration: float = 0.0
    events: list = field(default_factory=list)    # (type, time, progress, span_index, span_start)
    end_x: float = 0.0
    end_y: float = 0.0
    # spinner
    spins_required: int = 0
    max_bonus_spins: int = 0
    # processor results
    stack_height: int = 0
    combo_index: int = 0
    index_in_combo: int = 0
    last_in_combo: bool = False


@dataclass
class Beatmap:
    objects: list[HitObj]
    version: int
    cs: float
    ar: float
    od: float
    hp: float
    slider_multiplier: float
    tick_rate: float
    stack_leniency: float
    preempt: float
    fade_in: float
    radius: float
    scale: float
    windows: tuple[float, float, float]
    breaks: list

    @property
    def stack_offset_unit(self) -> float:
        return float(F(self.scale) * F(-6.4))


def _curve_position_at(o: HitObj, t: float) -> np.ndarray:
    """IHasPathWithRepeats.CurvePositionAt (t in [0,1] over the whole slider)."""
    p = t * o.span_count
    span = math.floor(p)
    frac = p - span
    if span >= o.span_count:   # t == 1
        span = o.span_count - 1
        frac = 1.0
    prog = 1 - frac if span % 2 == 1 else frac
    return o.path.position_at(prog)


def build_beatmap(osu: OsuFile) -> Beatmap:
    if osu.mode != 0:
        raise ValueError("not an osu!standard map")
    d = osu.difficulty
    hp = min(max(f32(d.get("HPDrainRate", 5)), 0), 10)
    cs = min(max(f32(d.get("CircleSize", 5)), 0), 10)
    od = min(max(f32(d.get("OverallDifficulty", 5)), 0), 10)
    ar = min(max(f32(d.get("ApproachRate", d.get("OverallDifficulty", 5))), 0), 10)
    sm = min(max(float(d.get("SliderMultiplier", 1.4)), 0.4), 3.6)
    tr = min(max(float(d.get("SliderTickRate", 1)), 0.5), 8.0)
    sl = f32(float(osu.general.get("StackLeniency", "0.7") or 0.7))
    version = osu.version
    offset = 24.0 if version < 5 else 0.0
    timing = Timing(osu)
    preempt = preempt_from_ar(ar)
    fade_in = fade_in_from_preempt(preempt)
    windows = hit_windows(od)
    radius = radius_from_cs(cs)
    scale = scale_from_cs(cs)

    objs: list[HitObj] = []
    prev_spinner = False
    for i, ro in enumerate(sorted(osu.hit_objects, key=lambda o: o.time)):
        t = ro.time + offset
        x, y = float(int(ro.x)), float(int(ro.y))
        if ro.type & 1:
            o = HitObj(CIRCLE, t, x, y, new_combo=(not objs) or prev_spinner or ro.new_combo,
                       combo_offset=ro.combo_skip if ro.new_combo else 0, end=t)
            prev_spinner = False
        elif ro.type & 2:
            o = HitObj(SLIDER, t, x, y, new_combo=(not objs) or prev_spinner or ro.new_combo,
                       combo_offset=ro.combo_skip if ro.new_combo else 0)
            exp = ro.length if (ro.length is not None and ro.length > 0) else None
            if exp is not None:
                exp = max(0.0, exp)
            control = convert_path(ro.curve_type, ro.curve_points, (x, y), version)
            o.path = build_path(control, exp)
            repeats = max(0, ro.slides - 1)
            if ro.slides > 9000:
                raise ValueError("too many slides")
            if abs(o.path.distance) <= 1e-7:
                repeats = 0
            o.span_count = repeats + 1
            beat = timing.beat_length_at(t)
            sv, gen = timing.difficulty_at(t)
            sv_beat = -100.0 / sv
            bpm_mult = float(min(max(F(-sv_beat), F(10)), F(1000))) / 100.0
            adj = beat * bpm_mult
            o.velocity = 100.0 * sm / adj
            scoring = o.velocity * beat
            tdm = (1.0 / sv) if version < 8 else 1.0
            o.tick_distance = (scoring / tr * tdm) if gen else math.inf
            o.end = t + o.span_count * o.path.distance / o.velocity
            o.span_duration = (o.end - t) / o.span_count
            prev_spinner = False
        elif ro.type & 8:
            end = max(0.0, (ro.end_time or t) + offset - t)
            o = HitObj(SPINNER, t, 256.0, 192.0, new_combo=ro.new_combo, end=t + end)
            prev_spinner = True
        else:
            continue
        objs.append(o)

    # --- slider nested events + end positions
    for o in objs:
        if o.kind != SLIDER:
            continue
        length = min(100000.0, o.path.distance)
        tick_d = min(max(o.tick_distance, 0.0), length) if math.isfinite(o.tick_distance) else length
        min_end = o.velocity * 10
        ev = []
        for span in range(o.span_count):
            span_start = o.start + span * o.span_duration
            rev = span % 2 == 1
            ticks = []
            if tick_d != 0:
                dd = tick_d
                while dd <= length:
                    if dd >= length - min_end:
                        break
                    p = dd / length
                    tp = 1 - p if rev else p
                    ticks.append((EV_TICK, span_start + tp * o.span_duration, p, span, span_start))
                    dd += tick_d
            if rev:
                ticks.reverse()
            ev.extend(ticks)
            if span < o.span_count - 1:
                ev.append((EV_REPEAT, o.start + (span + 1) * o.span_duration, float((span + 1) % 2), span, span_start))
        ev.append((EV_TAIL, o.start + o.span_count * o.span_duration, float(o.span_count % 2), o.span_count - 1, 0.0))
        ev.sort(key=lambda e: e[1])
        o.events = ev
        endp = _curve_position_at(o, 1.0)
        o.end_x, o.end_y = float(F(o.x) + endp[0]), float(F(o.y) + endp[1])
    for o in objs:
        if o.kind != SLIDER:
            o.end_x, o.end_y = o.x, o.y
        if o.kind == SPINNER:
            dur = o.end - o.start
            min_rps = difficulty_range(od, 90, 150, 225) / 60
            max_rps = difficulty_range(od, 250, 380, 430) / 60
            secs = dur / 1000
            o.spins_required = int(min_rps * secs + 0.0001)
            o.max_bonus_spins = max(0, int(max_rps * secs + 0.0001) - o.spins_required - 2)

    # --- breaks -> new combo
    breaks = [(a + offset, max(a, b) + offset) for a, b in osu.breaks]
    for bs, be in breaks:
        for o in objs:
            if be < o.start:
                o.new_combo = True
                break

    # --- combo indices (OsuHitObject.UpdateComboInformation)
    prev = None
    for idx, o in enumerate(objs):
        if prev is None:
            o.new_combo = True if o.kind != SPINNER else o.new_combo
        if o.kind != SPINNER and (o.new_combo or prev is None or prev.kind == SPINNER):
            o.index_in_combo = 0
            o.combo_index = (prev.combo_index if prev else 0) + 1
            if prev is not None:
                prev.last_in_combo = True
        elif prev is not None:
            o.index_in_combo = prev.index_in_combo + 1
            o.combo_index = prev.combo_index
        prev = o

    bm = Beatmap(objs, version, cs, ar, od, hp, sm, tr, sl, preempt, fade_in, radius, scale, windows, breaks)
    _apply_stacking(bm)
    return bm


def _dist(ax, ay, bx, by):
    dx, dy = F(ax) - F(bx), F(ay) - F(by)
    return float(F(math.sqrt(float(F(dx * dx + dy * dy)))))


def _apply_stacking(bm: Beatmap):
    h = bm.objects
    for o in h:
        o.stack_height = 0
    thr_of = lambda o: float(F(int(bm.preempt)) * F(bm.stack_leniency))
    if bm.version >= 6:
        n = len(h)
        for i in range(n - 1, 0, -1):
            obj_i = h[i]
            if obj_i.stack_height != 0 or obj_i.kind == SPINNER:
                continue
            thr = thr_of(obj_i)
            if obj_i.kind == CIRCLE:
                k = i
                while True:
                    k -= 1
                    if k < 0:
                        break
                    obj_n = h[k]
                    if obj_n.kind == SPINNER:
                        continue
                    if int(obj_i.start) - int(obj_n.end) > thr:
                        break
                    if obj_n.kind == SLIDER and _dist(obj_n.end_x, obj_n.end_y, obj_i.x, obj_i.y) < 3:
                        off = obj_i.stack_height - obj_n.stack_height + 1
                        for j in range(k + 1, i + 1):
                            if _dist(obj_n.end_x, obj_n.end_y, h[j].x, h[j].y) < 3:
                                h[j].stack_height -= off
                        break
                    if _dist(obj_n.x, obj_n.y, obj_i.x, obj_i.y) < 3:
                        obj_n.stack_height = obj_i.stack_height + 1
                        obj_i = obj_n
            elif obj_i.kind == SLIDER:
                k = i
                while True:
                    k -= 1
                    if k < 0:
                        break
                    obj_n = h[k]
                    if obj_n.kind == SPINNER:
                        continue
                    if obj_i.start - obj_n.start > thr:
                        break
                    if _dist(obj_n.end_x, obj_n.end_y, obj_i.x, obj_i.y) < 3:
                        obj_n.stack_height = obj_i.stack_height + 1
                        obj_i = obj_n
    else:
        n = len(h)
        for i in range(n):
            cur = h[i]
            if cur.stack_height != 0 and cur.kind != SLIDER:
                continue
            start_time = cur.end
            slider_stack = 0
            for j in range(i + 1, n):
                thr = thr_of(h[i])
                if h[j].start - thr > start_time:
                    break
                if cur.kind == SLIDER:
                    endp = cur.path.position_at(1.0)
                    p2x, p2y = float(F(cur.x) + endp[0]), float(F(cur.y) + endp[1])
                else:
                    p2x, p2y = cur.x, cur.y
                if _dist(h[j].x, h[j].y, cur.x, cur.y) < 3:
                    cur.stack_height += 1
                    start_time = h[j].start
                elif _dist(h[j].x, h[j].y, p2x, p2y) < 3:
                    slider_stack += 1
                    h[j].stack_height -= slider_stack
                    start_time = h[j].start


def stacked(bm: Beatmap, o: HitObj) -> tuple[float, float]:
    if o.kind == SPINNER:
        return o.x, o.y
    off = F(o.stack_height) * F(bm.scale) * F(-6.4)
    return float(F(o.x) + off), float(F(o.y) + off)


def max_combo(bm: Beatmap) -> int:
    n = 0
    for o in bm.objects:
        n += 1 if o.kind != SLIDER else 1 + len(o.events)
    return n
