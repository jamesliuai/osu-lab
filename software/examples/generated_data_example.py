"""Generated-data example for osu-sim (spec v3.3 section 11.4). CPU only; no community-authored map is used.

1. Generate three osu!standard maps with osu_sim.generator (fixed seeds, three curriculum levels) and write them as .osu.
2. Convert them (osu_sim.osufile + osu_sim.beatmap) and compile a MapPool (osu_sim.pool).
3. Play every map in parallel in SimEnv (60 Hz control, full map from time 0, no HP fail) with the scripted autopilot
   (osu_sim.autopilot; a diagnostic reference player, never used for training): once on time and once 45 ms late, the
   timing offset of the recorded "late" parity traces. Print the per-map judgement record.
4. Show the policy-facing protocol: the 8-dimensional aux vector and the render-state fields that the GPU renderer turns
   into the 256 x 192 RGB observation (rendering needs a GPU and the renderer assets; see docs/ENVIRONMENT.md).

Usage: python -B generated_data_example.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np


def play(pool, offset_ms):
    """All maps of the pool in parallel, one autopilot per slot; returns the env after every map has finished."""
    from osu_sim.autopilot import Autopilot
    from osu_sim.env import ClipConfig, SimEnv, UniformSampler
    env = SimEnv(pool, pool.n_maps, "cpu", seed=0, sampler=UniformSampler(np.arange(pool.n_maps)),
                 clip=ClipConfig(full_map=True, random_initial_cursor=False))
    for i in range(pool.n_maps):
        env._reset_one(i, i)
    pilots = [Autopilot(pool, env.dt, offset_ms=offset_ms, rng=np.random.default_rng(i)) for i in range(pool.n_maps)]
    done = np.zeros(pool.n_maps, bool)
    steps = np.zeros(pool.n_maps, int)
    first = True
    while not done.all():
        if first:
            print(f"aux vector at the first step, slot 0: {np.round(env.aux_np()[0], 3).tolist()} "
                  "[previous cursor command x, y; one-hot held keys 0..3; 0 on the first observation else 1; 0]")
            print(f"render-state fields passed to the renderer: {sorted(env.render_state_np())}")
            first = False
        acts, keys = zip(*(pilots[i].act(env.e_t[i], int(env.e_lo[i]), int(env.e_hi[i]), int(env.e_keys[i]))
                           for i in range(pool.n_maps)))
        _, d, _ = env.step_np(np.array(acts), np.array(keys))
        steps += ~done
        done |= d
    return env, steps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None, help="where to write the .osu files (default: a new temporary directory)")
    a = ap.parse_args(argv)
    from osu_sim import rules
    from osu_sim.beatmap import build_beatmap
    from osu_sim.generator import generate, sample_params
    from osu_sim.osufile import parse_osu, write_osu
    from osu_sim.pool import compile_pool

    out = Path(a.out_dir) if a.out_dir else Path(tempfile.mkdtemp(prefix="osu_sim_example_"))
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for seed, level in ((11, 0.2), (12, 0.5), (13, 0.8)):
        rng = np.random.default_rng(seed)
        path = out / f"example_seed{seed}.osu"
        write_osu(generate(rng, sample_params(level, rng)), path)
        bm = build_beatmap(parse_osu(path))
        kinds = [o.kind for o in bm.objects]
        print(f"{path.name}: level {level}, {len(bm.objects)} objects (circles {kinds.count(0)}, sliders "
              f"{kinds.count(1)}, spinners {kinds.count(2)}), OD {bm.od:g}, CS {bm.cs:g}, AR {bm.ar:g}, "
              f"last object ends at {max(o.end for o in bm.objects) / 1000:.1f} s")
        items.append((path.name, bm))
    pool = compile_pool(items)
    for label, offset in (("autopilot on time", 0.0), ("autopilot 45 ms late", 45.0)):
        print(f"\n{label}:")
        env, steps = play(pool, offset)
        for i in range(pool.n_maps):
            s = env.e_stats[i]
            fails = int(s[rules.ST_BREAKS] + s[rules.ST_TAIL_MISS])
            print(f"  {pool.keys[i]}: steps {steps[i]}, Great {int(s[rules.ST_GREAT])} Ok {int(s[rules.ST_OK])} "
                  f"Meh {int(s[rules.ST_MEH])} Miss {int(s[rules.ST_MISS])}; ticks {int(s[rules.ST_TICK_HIT])}/"
                  f"{int(s[rules.ST_TICK_HIT] + s[rules.ST_TICK_MISS])}; tails {int(s[rules.ST_TAIL_HIT])}/"
                  f"{int(s[rules.ST_TAIL_HIT] + s[rules.ST_TAIL_MISS])}; spinners {int(s[rules.ST_SPIN_HIT])}/"
                  f"{int(s[rules.ST_SPIN_HIT] + s[rules.ST_SPIN_MISS])}; failures (breaks + missed tails) {fails}; "
                  f"max combo {int(env.e_combo[i, 1])}; game reward {s[rules.ST_REWARD]:.2f}")
    print(f"\n.osu files written to {out}")


if __name__ == "__main__":
    main()
