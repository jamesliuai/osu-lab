"""Study PD generated-pool builder (NEXT_MILESTONE_PD_PROMPT.md section 5).

Copied from packaging/osu_sim/examples/recipe/build_recipe_pools.py (gen_one's map construction, save_compressed,
sha256_file; the original is unchanged) and changed for Study PD:

- The variant (slider drill / spinner drill / stream-jump drill / plain) is assigned explicitly, not by seed % 20.
  The construction per map is unchanged: sample_params(level, default_rng(seed)), the variant override, generate().
- Band and variant per sequence position come from sequential largest-remainder quotas (quota_sequence), so every
  prefix of a sequence holds both share vectors within +-1 map. Bands are the recipe mixture's pseudo-star ranges
  [0, 3), [3, 4.5), [4.5, 5.5) = levels [0, 5/9), [5/9, 8/9), [8/9, 1] with the mixture weights renormalised without
  the empty drill range (0.2062 / 0.3505 / 0.4433); variant shares 0.25 / 0.15 / 0.10 / 0.50 as the recipe builder.
- Level: uniform inside the band with stratified jitter (base-2 van der Corput point + jitter inside a 2^-12 stratum,
  so the first 2^m maps of a band fill its 2^m strata), kept 1e-5 of the band width away from the band edges.
- Replicate r (r = 0 pilot only, r = 1-4 study): sequence position i = 0..8191, seed 2,000,000 + 100,000 r + i.
  Pool N = the first N positions (nested). An empty map is replaced from the spare seeds 2,050,000 + 100,000 r + j
  (same position, band, variant and level) and logged.
- Monitor pd_monitor_v1: 200 maps of the same construction, seeds 2,900,000 + k (spares 2,950,000 + j). Not sealed;
  learning curves and the pilot gate only; never trained on.
- Shift sets (P3 only): tempo = the upper two bands, same variant mix, prm.bpm *= 1.3 after the variant override;
  pattern = generate_burst_drill with default BurstParams, placeholder pseudo-stars U[21, 21.5), source 2, level -1
  (the convention of scripts/sim_build_drill_pool.py).
- Sealed tests (P3 only): seed 5,000,000 + (s mod 4,000,000) + j with s from the two RFC 3161 tokens; j = 0..499 T_ID,
  500..749 T_tempo, 750..999 T_pattern; spares from j = 1000 on. Prints counts only.

Pool fields: maps["stars"] = 0.5 + 4.5 * level (pseudo difficulty, not a star rating), maps["source"] = 1 (2 for burst
drills), maps["level"]. Pools are written compressed and exclusively ("xb"); one manifest per pool records every map's
seed, level, band, variant and .osu sha256, the osu_sim source hashes, the pool's file sha256 and a content sha256 over
its arrays (the zip container carries a write time, so a rebuild is compared by content, not by file hash).

Portable release derivative: the construction functions are preserved, while all output locations are explicit.
No source-repository paths are inferred. The output directory must not exist. The command line supports training
and monitor pools only; sealed generation and timestamp-token file I/O are omitted. Pure construction helpers for
shift families remain available for inspection but are not executed by these commands.

Usage (after installing the exported package):
    python -B tools/build_pd_pools.py train --out-root generated_pd_train --replicate 1 --sizes 16,256,4096
    python -B tools/build_pd_pools.py monitor --out-root generated_pd_monitor

Even when writing smaller prefixes, train constructs the full 8,192-position sequence to preserve its original
construction and empty-map replacement semantics. The default is one CPU worker. Outputs are generated practice
material, not a published study result. Newly written manifests use relative paths and include source hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


MIX_W = (0.2, 0.34, 0.43)                                    # pkg_recipe_r1 env.mixture weights of the 3 map ranges
BAND_SHARES = tuple(w / sum(MIX_W) for w in MIX_W)           # the drill range [20, 20.5) is empty in generated pools
BAND_LEVELS = ((0.0, 5 / 9), (5 / 9, 8 / 9), (8 / 9, 1.0))
BAND_STARS = ((0.0, 3.0), (3.0, 4.5), (4.5, 5.5))
VARIANTS = ("slider", "spinner", "streamjump", "plain")
VARIANT_SHARES = (0.25, 0.15, 0.10, 0.50)
N_SEQ = 8192
TRAIN_SEED0, REP_STRIDE, SPARE_OFFSET = 2_000_000, 100_000, 50_000
MONITOR_SEED0, MONITOR_SPARE0, MONITOR_N = 2_900_000, 2_950_000, 200
TEST_SEED0, TEST_MOD, TEST_SPARE_J0 = 5_000_000, 4_000_000, 1000
TEST_SETS = (("id", 500), ("tempo", 250), ("pattern", 250))
TEMPO_FACTOR = 1.3
EDGE = 1e-5                                                  # level margin from the band edges (fraction of width)
SOURCES = ("generator.py", "osufile.py", "beatmap.py", "pool.py", "difficulty.py")


def quota_sequence(shares, n):
    """Category per position by sequential largest-remainder quotas: at prefix length t pick the category with the
    largest remainder shares[c] * t - count[c] (ties: lowest index). Every prefix holds each category within +-1."""
    shares = np.asarray(shares, dtype=np.float64)
    if abs(shares.sum() - 1.0) > 1e-9:
        raise ValueError(f"shares must sum to 1: {shares}")
    counts = np.zeros(len(shares), dtype=np.int64)
    out = np.empty(n, dtype=np.int64)
    for t in range(1, n + 1):
        c = int(np.argmax(shares * t - counts))
        counts[c] += 1
        out[t - 1] = c
    return out


def van_der_corput(k):
    x, denom = 0.0, 1.0
    while k:
        denom *= 2.0
        k, bit = divmod(k, 2)
        x += bit / denom
    return x


def stratified_levels(bands, band_levels, rng_seed):
    """Level per position: the k-th position of band b gets lo + (hi - lo) * (EDGE + (1 - 2 EDGE) * x_k) with
    x_k = vdc2(k) + u_k * 2^-12, u_k ~ U(0, 1) from default_rng([*rng_seed, b])."""
    levels = np.empty(len(bands), dtype=np.float64)
    for b, (lo, hi) in enumerate(band_levels):
        idx = np.nonzero(bands == b)[0]
        u = np.random.default_rng([*rng_seed, b]).random(len(idx))
        x = np.array([van_der_corput(k) for k in range(len(idx))]) + u * 2.0 ** -12
        levels[idx] = lo + (hi - lo) * (EDGE + (1 - 2 * EDGE) * x)
    return levels


def practice_sequence(n, rng_seed, band_shares=BAND_SHARES, band_ids=(0, 1, 2)):
    """(band, variant, level) per position of a practice-construction sequence of length n."""
    bands = np.asarray(band_ids)[quota_sequence(band_shares, n)]
    variants = quota_sequence(VARIANT_SHARES, n)
    levels = stratified_levels(bands, BAND_LEVELS, rng_seed)
    return bands, variants, levels


def gen_one(job):
    """job: dict(seed, kind, level, variant). kind 'practice' / 'tempo' (bpm x 1.3 after the variant override) /
    'pattern' (generate_burst_drill, default BurstParams). Returns (n_objects, beatmap, osu sha256)."""
    from osu_sim.beatmap import build_beatmap
    from osu_sim.generator import BurstParams, generate, generate_burst_drill, sample_params
    from osu_sim.osufile import parse_osu, write_osu
    rng = np.random.default_rng(job["seed"])
    if job["kind"] == "pattern":
        text = write_osu(generate_burst_drill(rng, BurstParams()))
    else:
        level, variant = job["level"], job["variant"]
        prm = sample_params(level, rng)
        if variant == "slider":
            prm.slider_prob, prm.stream_prob = 0.75, 0.0
            prm.slider_repeat_prob = 0.3
        elif variant == "spinner":
            prm.spinner_prob, prm.slider_prob, prm.stream_prob = 0.3, 0.2, 0.0
            prm.spinner_beats = (2.0, 8.0)
        elif variant == "streamjump":
            prm.stream_prob = 0.35 if level > 0.2 else 0.0
            prm.spacing = (prm.spacing[0], prm.spacing[1] * 1.3)
        elif variant != "plain":
            raise ValueError(variant)
        if job["kind"] == "tempo":
            prm.bpm *= TEMPO_FACTOR
        elif job["kind"] != "practice":
            raise ValueError(job["kind"])
        text = write_osu(generate(rng, prm))
    bm = build_beatmap(parse_osu(text))
    return len(bm.objects), bm, hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_all(jobs, spare_seeds, workers, log):
    """Generate jobs in order; an empty map is regenerated (same slot) with the next spare seed, sequentially."""
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(gen_one, jobs, chunksize=16))
    spare = iter(spare_seeds)
    for i, (n_obj, _, _) in enumerate(res):
        while n_obj == 0:
            old = jobs[i]["seed"]
            jobs[i] = dict(jobs[i], seed=next(spare), replaced_seed=old)
            n_obj, bm, h = gen_one(jobs[i])
            res[i] = (n_obj, bm, h)
            log.append(dict(position=i, empty_seed=old, spare_seed=jobs[i]["seed"]))
    return res


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


def pool_content_sha256(pool):
    """sha256 over the pool's tables (name, dtype, shape, bytes; fields sorted), keys and breaks."""
    h = hashlib.sha256()
    for name in ("maps", "objs", "nest", "path"):
        table = getattr(pool, name)
        for k in sorted(table):
            v = np.ascontiguousarray(table[k])
            h.update(f"{name}.{k}:{v.dtype.str}:{v.shape}".encode())
            h.update(v.tobytes())
    h.update("\n".join(pool.keys).encode("utf-8"))
    for b in pool.breaks:
        a = np.asarray(b, dtype=np.float64).reshape(-1, 2)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def compile_maps(items):
    """items: [dict(key, beatmap, stars, source, level)] -> MapPool with stars / source / level fields."""
    from osu_sim.pool import compile_pool
    pool = compile_pool([(it["key"], it["beatmap"]) for it in items])
    pool.maps["stars"] = np.array([it["stars"] for it in items], dtype=np.float32)
    pool.maps["source"] = np.array([it["source"] for it in items], dtype=np.int64)
    pool.maps["level"] = np.array([it["level"] for it in items], dtype=np.float32)
    return pool


def source_hashes():
    import osu_sim
    src = Path(osu_sim.__file__).resolve().parent
    return {n: sha256_file(src / n) for n in SOURCES}


def check_star_bands(items):
    """Every practice / tempo map's float32 pseudo-stars lie inside its band's star range."""
    for it in items:
        if it["source"] != 1:
            continue
        lo, hi = BAND_STARS[it["band"]]
        s = np.float32(it["stars"])
        if not (lo <= s < hi):
            raise SystemExit(f"{it['key']}: stars {s} outside band {it['band']} [{lo}, {hi})")


def practice_items(seq_seeds, bands, variants, levels, kind, res, key_fmt):
    items = []
    for i, ((n_obj, bm, h), seed) in enumerate(zip(res, seq_seeds)):
        lv = float(levels[i])
        items.append(dict(key=key_fmt.format(i=i, seed=seed), beatmap=bm, stars=0.5 + 4.5 * lv, source=1, level=lv,
                          band=int(bands[i]), variant=VARIANTS[int(variants[i])], seed=int(seed), osu_sha256=h,
                          n_objects=n_obj, kind=kind))
    return items


def write_pool(items, path, manifest_path, meta):
    if path.exists() or manifest_path.exists():
        raise SystemExit(f"refusing to overwrite {path} or {manifest_path}")
    pool = compile_maps(items)
    save_compressed(pool, path)
    from osu_sim.pool import MapPool
    back = MapPool.load(path)
    content = pool_content_sha256(pool)
    if pool_content_sha256(back) != content:
        raise SystemExit(f"{path}: reloaded pool differs from the compiled pool")
    src_hashes = source_hashes()
    entry = dict(meta, path=path.name, sha256=sha256_file(path), content_sha256=content,
                 n_maps=pool.n_maps, n_objects=int(len(pool.objs["kind"])),
                 builder=Path(__file__).name, builder_sha256=sha256_file(Path(__file__)),
                 osu_sim_sources=src_hashes,
                 stars_note="maps['stars'] = 0.5 + 4.5 * level (pseudo difficulty, not a star rating); burst drills "
                            "U[21, 21.5), source 2, level -1",
                 maps=[{k: it[k] for k in ("key", "seed", "level", "band", "variant", "kind", "osu_sha256",
                                           "n_objects") if k in it} | ({"set": it["set"]} if "set" in it else {})
                       for it in items])
    with open(manifest_path, "x", encoding="utf-8", newline="\n") as f:
        json.dump(entry, f, indent=1)
    return entry


def build_train(r, sizes, workers, out_dir):
    pool_dir = Path(out_dir)
    t0 = time.perf_counter()
    bands, variants, levels = practice_sequence(N_SEQ, [20261101, r])
    base = TRAIN_SEED0 + REP_STRIDE * r
    jobs = [dict(seed=base + i, kind="practice", level=float(levels[i]), variant=VARIANTS[int(variants[i])])
            for i in range(N_SEQ)]
    log = []
    res = generate_all(jobs, range(base + SPARE_OFFSET, base + REP_STRIDE), workers, log)
    seeds = [j["seed"] for j in jobs]
    items = practice_items(seeds, bands, variants, levels, "practice", res, "pd:r%d:{i}:{seed}" % r)
    check_star_bands(items)
    print(f"replicate {r}: generated {len(items)} maps in {time.perf_counter() - t0:.0f} s; empty replaced: {log}",
          flush=True)
    pool_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for n in sorted(sizes):
        name = f"pd_r{r}_n{n}"
        meta = dict(pool=name, replicate=r, n=n, construction="practice", seed_base=base,
                    spare_seed_range=[base + SPARE_OFFSET, base + REP_STRIDE - 1], empty_replaced=log,
                    band_shares=BAND_SHARES, band_levels=BAND_LEVELS, variant_shares=VARIANT_SHARES,
                    level_rng_seed=[20261101, r])
        e = write_pool(items[:n], pool_dir / f"{name}.npz", pool_dir / f"{name}.manifest.json", meta)
        out[name] = e
        print(f"{name}: {e['n_maps']} maps, {e['n_objects']} objects, sha256 {e['sha256']}, content "
              f"{e['content_sha256'][:16]} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return out


def build_monitor(workers, out_dir):
    pool_dir = Path(out_dir)
    bands, variants, levels = practice_sequence(MONITOR_N, [20261101, 99])
    jobs = [dict(seed=MONITOR_SEED0 + k, kind="practice", level=float(levels[k]), variant=VARIANTS[int(variants[k])])
            for k in range(MONITOR_N)]
    log = []
    res = generate_all(jobs, range(MONITOR_SPARE0, MONITOR_SPARE0 + 50_000), workers, log)
    items = practice_items([j["seed"] for j in jobs], bands, variants, levels, "practice", res, "pdmon:{i}:{seed}")
    check_star_bands(items)
    pool_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(pool="pd_monitor_v1", construction="practice", seed_base=MONITOR_SEED0,
                spare_seed_range=[MONITOR_SPARE0, MONITOR_SPARE0 + 49_999], empty_replaced=log,
                band_shares=BAND_SHARES, band_levels=BAND_LEVELS, variant_shares=VARIANT_SHARES,
                level_rng_seed=[20261101, 99], note="not sealed; learning curves and the pilot gate only; never trained on")
    e = write_pool(items, pool_dir / "pd_monitor_v1.npz", pool_dir / "pd_monitor_v1.manifest.json", meta)
    print(f"pd_monitor_v1: {e['n_maps']} maps, {e['n_objects']} objects, sha256 {e['sha256']}", flush=True)
    return e


def token_seed(tsr_a: bytes, tsr_b: bytes) -> int:
    return int.from_bytes(hashlib.sha256(tsr_a + tsr_b).digest()[:8], "big")


def test_seed(s, j):
    return TEST_SEED0 + (s % TEST_MOD) + j


def sealed_items(s, workers):
    """The three sealed sets' items (in T_ID, T_tempo, T_pattern order) from token seed s; spares from j = 1000."""
    spare = (test_seed(s, j) for j in range(TEST_SPARE_J0, TEST_SPARE_J0 + 10_000))
    log, out, j0 = [], {}, 0
    for name, n in TEST_SETS:
        seeds = [test_seed(s, j0 + j) for j in range(n)]
        if name == "pattern":
            jobs = [dict(seed=sd, kind="pattern") for sd in seeds]
            res = generate_all(jobs, spare, workers, log)
            stars_rng = np.random.default_rng([20261101, 21])
            st = (21.0 + 0.5 * stars_rng.random(n)).astype(np.float32)
            st = np.minimum(st, np.nextafter(np.float32(21.5), np.float32(21.0)))
            items = [dict(key=f"pdtest:pattern:{k}:{jobs[k]['seed']}", beatmap=bm, stars=float(st[k]), source=2,
                          level=-1.0, band=-1, variant="burst", seed=int(jobs[k]["seed"]), osu_sha256=h,
                          n_objects=n_obj, kind="pattern", set="pattern")
                     for k, (n_obj, bm, h) in enumerate(res)]
        else:
            if name == "id":
                bands, variants, levels = practice_sequence(n, [20261101, 7, s % TEST_MOD])
                kind = "practice"
            else:
                upper = BAND_SHARES[1] + BAND_SHARES[2]             # the upper two bands, renormalised
                shares = (BAND_SHARES[1] / upper, BAND_SHARES[2] / upper)
                bands, variants, levels = practice_sequence(n, [20261101, 8, s % TEST_MOD], band_shares=shares,
                                                            band_ids=(1, 2))
                kind = "tempo"
            jobs = [dict(seed=sd, kind=kind, level=float(levels[k]), variant=VARIANTS[int(variants[k])])
                    for k, sd in enumerate(seeds)]
            res = generate_all(jobs, spare, workers, log)
            items = practice_items([jb["seed"] for jb in jobs], bands, variants, levels, kind, res,
                                   "pdtest:%s:{i}:{seed}" % name)
            for it in items:
                it["set"] = name
            check_star_bands(items)
        out[name] = items
        j0 += n
    return out, log




def disjointness(sets, known):
    """Overlap counts of the sealed sets' .osu hashes with every known pool and with each other (all must be 0)."""
    hs = {name: {it["osu_sha256"] for it in items} for name, items in sets.items()}
    rep = {f"{name} & {pool}": len(h & k) for name, h in hs.items() for pool, k in known.items()}
    names = list(hs)
    rep.update({f"{x} & {y}": len(hs[x] & hs[y]) for i, x in enumerate(names) for y in names[i + 1:]})
    rep.update({f"{name} internal duplicates": len(items) - len(hs[name]) for name, items in sets.items()})
    return rep




def main(argv=None):
    ap = argparse.ArgumentParser(description="Portable procedural practice-pool builder; no sealed-data command.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--replicate", type=int, required=True)
    t.add_argument("--sizes", default="16,128,256,1024,8192")
    m = sub.add_parser("monitor")
    for p in (t, m):
        p.add_argument("--out-root", required=True, help="new output directory; must not already exist")
        p.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    if a.workers < 1:
        ap.error("--workers must be positive")
    if a.cmd == "train":
        if not 0 <= a.replicate <= 4:
            ap.error("replicate must be 0-4")
        try:
            sizes = [int(x) for x in a.sizes.split(",")]
        except ValueError:
            ap.error("--sizes must be comma-separated integers")
        if not sizes or len(set(sizes)) != len(sizes) or any(n < 1 or n > N_SEQ for n in sizes):
            ap.error("--sizes must contain distinct positive prefixes no larger than 8192")
    out = Path(a.out_root).resolve()
    if out.exists():
        ap.error("--out-root must be a new directory")
    out.mkdir(parents=True, exist_ok=False)
    if a.cmd == "train":
        build_train(a.replicate, sizes, a.workers, out)
    else:
        build_monitor(a.workers, out)


if __name__ == "__main__":
    main()
