"""Build the generated-only pools of the osu-sim reproducible recipe (spec v3.3 section 11.4).

Every map is made by osu_sim.generator from a fixed seed, so the pools contain no community-authored content and are
reproducible from this package alone. The construction follows the project's generated-curriculum builder
(scripts/sim_build_gen_pool.py, gen_one): per map a seed and a curriculum level in [0, 1]; seed % 20 picks the variant
(0-4 slider drill, 5-7 spinner drill, 8-9 stream / jump drill, else the plain level parameters); the map's pseudo
difficulty is maps["stars"] = 0.5 + 4.5 * level (not a native star rating) and maps["source"] = 1 (generated).

    training pool:   n_train maps, seeds train_seed0 + i, levels = U(0, 1) ** 1.3 drawn from default_rng(levels_seed)
    evaluation pool: n_eval maps, seeds eval_seed0 + k, levels (k + 0.5) / n_eval (evenly spaced), disjoint seeds

Pools are saved compressed (np.savez_compressed; MapPool.load reads them). A manifest records every parameter, the
osu_sim source hashes of the modules used and the sha256 of both pools.

Usage (after installing this package):
    python -B examples/recipe/build_recipe_pools.py --out-dir generated_pools --workers 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DEFAULTS = dict(n_train=2000, n_eval=40, train_seed0=700000, eval_seed0=950000, levels_seed=20261022)


def gen_one(args):
    seed, level = args
    from osu_sim.beatmap import build_beatmap
    from osu_sim.generator import generate, sample_params
    from osu_sim.osufile import parse_osu, write_osu
    rng = np.random.default_rng(seed)
    prm = sample_params(level, rng)
    variant = seed % 20
    if variant < 5:          # slider drill
        prm.slider_prob, prm.stream_prob = 0.75, 0.0
        prm.slider_repeat_prob = 0.3
    elif variant < 8:        # spinner drill
        prm.spinner_prob, prm.slider_prob, prm.stream_prob = 0.3, 0.2, 0.0
        prm.spinner_beats = (2.0, 8.0)
    elif variant < 10:       # stream / jump drill
        prm.stream_prob = 0.35 if level > 0.2 else 0.0
        prm.spacing = (prm.spacing[0], prm.spacing[1] * 1.3)
    text = write_osu(generate(rng, prm))
    return f"gen:{seed}:{level:.3f}", build_beatmap(parse_osu(text)), float(level), hashlib.sha256(
        text.encode("utf-8")).hexdigest()


def build(jobs, workers):
    from osu_sim.pool import compile_pool
    with ProcessPoolExecutor(workers) as ex:
        gens = list(ex.map(gen_one, jobs, chunksize=16))
    kept = [(k, b, lv, h) for k, b, lv, h in gens if b.objects]
    pool = compile_pool([(k, b) for k, b, _, _ in kept])
    pool.maps["stars"] = np.array([0.5 + 4.5 * lv for _, _, lv, _ in kept], dtype=np.float32)
    pool.maps["source"] = np.ones(len(kept), dtype=np.int64)
    pool.maps["level"] = np.array([lv for _, _, lv, _ in kept], dtype=np.float32)
    return pool, [dict(key=k, level=lv, osu_sha256=h) for k, _, lv, h in kept], len(gens) - len(kept)


def save_compressed(pool, path):
    arrays = {}
    for name, table in (("m", pool.maps), ("o", pool.objs), ("n", pool.nest), ("p", pool.path)):
        for k, v in table.items():
            arrays[f"{name}_{k}"] = v
    arrays["keys"] = np.array(pool.keys, dtype=object)
    arrays["breaks"] = np.array([np.array(b, dtype=np.float64).reshape(-1, 2) for b in pool.breaks] +
                                [np.zeros((0, 2))], dtype=object)[:-1]
    with open(path, "xb") as f:                       # never overwrite an existing pool
        np.savez_compressed(f, **arrays, allow_pickle=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=v)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    levels = np.random.default_rng(a.levels_seed).uniform(0, 1, a.n_train) ** 1.3
    train_jobs = [(a.train_seed0 + i, float(levels[i])) for i in range(a.n_train)]
    eval_jobs = [(a.eval_seed0 + k, (k + 0.5) / a.n_eval) for k in range(a.n_eval)]
    if {s for s, _ in train_jobs} & {s for s, _ in eval_jobs}:
        raise SystemExit("training and evaluation seeds overlap")
    import osu_sim
    src = Path(osu_sim.__file__).resolve().parent
    manifest = dict(tool="build_recipe_pools.py", params={k: getattr(a, k) for k in DEFAULTS},
                    osu_sim_sources={n: sha256_file(src / n) for n in ("generator.py", "osufile.py", "beatmap.py",
                                                                       "pool.py", "difficulty.py")},
                    stars_note="maps['stars'] = 0.5 + 4.5 * level (pseudo difficulty, not a native star rating)",
                    pools={})
    for name, jobs in (("recipe_train_v1", train_jobs), ("recipe_eval_v1", eval_jobs)):
        pool, maps, dropped = build(jobs, a.workers)
        path = out / f"{name}.npz"
        save_compressed(pool, path)
        manifest["pools"][name] = dict(path=path.name, sha256=sha256_file(path), n_maps=pool.n_maps,
                                       n_objects=int(len(pool.objs["kind"])), dropped_empty=dropped,
                                       stars_min=float(pool.maps["stars"].min()),
                                       stars_max=float(pool.maps["stars"].max()), maps=maps)
        print(f"{name}: {pool.n_maps} maps, {len(pool.objs['kind'])} objects -> {path}", flush=True)
    with open(out / "recipe_pools_manifest.json", "x", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "maps"} for k, v in manifest["pools"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
