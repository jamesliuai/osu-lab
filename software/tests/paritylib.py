"""Simulator side of the recorded parity cases (a copy of the sim half of scripts/sim_parity.py; no native host).

scripts/sim_parity.py produced each recorded case by (1) building a scripted open-loop trace in osu-sim with the
autopilot (variants perfect / sloppy / late, seeded), (2) replaying that action trace step by step in the native
osu!lazer host, and (3) comparing per-step judgement counts. This module repeats step (1) with the installed package so
that its judgement totals, max combo and step count can be compared with the native values recorded in step (2).
The map generation recipe of the parity runs is reproduced by regenerate_parity_maps().
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

VARIANTS = dict(perfect={}, sloppy=dict(offset_ms=0.0, miss_prob=0.08), late=dict(offset_ms=45.0),
                early=dict(offset_ms=-60.0, miss_prob=0.03))


def sim2native():
    from osu_sim import rules
    names = {rules.ST_GREAT: "Great", rules.ST_OK: "Ok", rules.ST_MEH: "Meh", rules.ST_MISS: "Miss",
             rules.ST_TICK_HIT: "LargeTickHit", rules.ST_TICK_MISS: "LargeTickMiss",
             rules.ST_TAIL_HIT: "SliderTailHit", rules.ST_TAIL_MISS: "IgnoreMiss",
             rules.ST_SPIN_HIT: "spin_hit", rules.ST_SPIN_MISS: "spin_miss"}
    return names


def sim_trace(osu_path: Path, variant: str, seed: int, rate: float = 1.0):
    """-> (per-step judgement dicts, max combo, steps, timeout), exactly as scripts/sim_parity.py sim_trace."""
    from osu_sim.autopilot import Autopilot
    from osu_sim.beatmap import build_beatmap
    from osu_sim.env import ClipConfig, SimEnv, UniformSampler
    from osu_sim.osufile import parse_osu
    from osu_sim.pool import compile_pool
    bm = build_beatmap(parse_osu(Path(osu_path)))
    pool = compile_pool([(str(osu_path), bm)])
    if rate == 1.0:
        env = SimEnv(pool, 1, "cpu", seed=seed, sampler=UniformSampler([0]),
                     clip=ClipConfig(full_map=True, random_initial_cursor=False))
    else:
        from osu_sim.rate import MapRateSimEnv
        pool.maps["clock_rate"] = np.full(pool.n_maps, float(rate), dtype=np.float64)
        env = MapRateSimEnv(pool, 1, "cpu", seed=seed, sampler=UniformSampler([0]), control_hz=60,
                            clip=ClipConfig(full_map=True, random_initial_cursor=False))
    env._reset_one(0, 0)
    pilot = Autopilot(pool, env.dt, rng=np.random.default_rng(seed), **VARIANTS[variant])
    if variant == "sloppy":
        pilot.spin_deg = float(np.random.default_rng(seed).choice([20.0, 35.0, 60.0, 179.0, 200.0]))
    rng = np.random.default_rng(seed + 7)
    names = sim2native()
    per_step = []
    prev = env.e_stats[0].copy()
    while True:
        a, k = pilot.act(env.e_t[0], int(env.e_lo[0]), int(env.e_hi[0]), int(env.e_keys[0]))
        if variant == "sloppy":
            a = np.clip(a + rng.normal(0, 0.01, 2), -1, 1)
        _, done, timeout = env.step_np(np.array([a]), np.array([k]))
        cur = env.e_stats[0].copy()
        d = cur - prev
        prev = cur
        c = Counter()
        for i, name in names.items():
            if d[i]:
                c[name] += int(round(d[i]))
        per_step.append(dict(c))
        if done[0]:
            break
    return per_step, int(env.e_combo[0, 1]), len(per_step), bool(timeout[0])


def totals(per_step):
    t = Counter()
    for s in per_step:
        t.update(s)
    return dict(t)


def regenerate_parity_maps(gen_maps: int = 4, spinner_maps: int = 0):
    """The generated maps of a scripts/sim_parity.py run: default_rng(5); gen i = generate(rng, sample_params(0.3 +
    0.7 * i / (gen_maps - 1), rng)); then spin i with spinner-drill parameters. -> {name: .osu text}."""
    from osu_sim.generator import generate, sample_params
    from osu_sim.osufile import write_osu
    rng = np.random.default_rng(5)
    out = {}
    for i in range(gen_maps):
        f = generate(rng, sample_params(0.3 + 0.7 * i / max(1, gen_maps - 1), rng))
        out[f"gen{i}"] = write_osu(f)
    for i in range(spinner_maps):
        prm = sample_params(0.5, rng)
        prm.spinner_prob, prm.slider_prob, prm.stream_prob = 0.35, 0.15, 0.0
        prm.spinner_beats = (1.0 + i, 3.0 + 2 * i)
        prm.od = [2.0, 5.0, 8.0, 10.0][i % 4]
        out[f"spin{i}"] = write_osu(generate(rng, prm))
    return out
