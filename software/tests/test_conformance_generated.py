"""osu-sim conformance tests on generated content (spec v3.3 section 11.4).

Data: tests/data (built by tools/collect_conformance_data.py): the 10 generated maps gen0-3 and spin0-5 and the 72
native parity records of those maps from six recorded directories (rate 1.0: parity2, parity_spin; rate 1.0 through
the host's rate path: b3_rate1_parity2, b3_rate1_spin; rate 1.5 = native Double Time: b4_dt15_parity2, b4_dt15_spin).
No community-authored map is used.

1. Generation: osu_sim.generator reproduces the recorded maps byte for byte (line endings normalised).
2. Conversion: osu_sim's beatmap conversion of each map has the object structure the native judgement totals imply
   (heads, slider tails, ticks / repeats, spinners) and, where the native trace has no miss, the native max combo.
3. Judgement parity: the scripted trace re-simulated with the installed package gives exactly the recorded native
   per-type judgement totals, max combo and step count (and the recorded simulator totals) in every case. The per-step
   native events were compared at recording time (0 mismatched steps in every case, asserted from the records) but
   were not retained, so this test checks totals, combo and step count, not each step.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pytest

import paritylib

DATA = Path(__file__).resolve().parent / "data"
MANIFEST = json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))
CASES = MANIFEST["cases"]


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n")


def test_installed_package_location():
    import importlib.util
    import osu_sim
    want = os.environ.get("OSU_SIM_EXPECT_UNDER")
    if not want:
        pytest.skip("OSU_SIM_EXPECT_UNDER not set (only the offline install check sets it)")
    import sys
    import osu_rl.policy.priors  # noqa: F401  (the vendored module the S1 head imports)
    root = Path(want).resolve()
    here = Path(osu_sim.__file__).resolve()
    assert root in here.parents, f"osu_sim imported from {here}, not under {want}"
    # every loaded osu_sim / osu_rl module must come from the installation (a source checkout on sys.path, e.g. an
    # editable .pth of the project environment, must not supply any of them)
    for name, mod in list(sys.modules.items()):
        if name.split(".")[0] in ("osu_sim", "osu_rl") and getattr(mod, "__file__", None):
            assert root in Path(mod.__file__).resolve().parents, f"{name} loaded from {mod.__file__}"
    assert importlib.util.find_spec("osu_sim.realhost") is None, "the host driver must not be in the distribution"
    assert importlib.util.find_spec("osu_rl.env") is None, "only osu_rl.policy.priors is vendored"


def test_data_files_match_manifest():
    for name, rec in MANIFEST["maps"].items():
        assert hashlib.sha256((DATA / "maps" / f"{name}.osu").read_bytes()).hexdigest() == rec["sha256"]
    for c in CASES:
        assert hashlib.sha256((DATA / c["copy"]).read_bytes()).hexdigest() == c["sha256"]
    assert len(MANIFEST["maps"]) == 10 and len(CASES) == 72
    assert not any(c["map"].startswith("real") for c in CASES)


@pytest.mark.parametrize("gen_maps,spinner_maps,names", [(4, 0, [f"gen{i}" for i in range(4)]),
                                                         (0, 6, [f"spin{i}" for i in range(6)])])
def test_generator_reproduces_recorded_maps(gen_maps, spinner_maps, names):
    regen = paritylib.regenerate_parity_maps(gen_maps, spinner_maps)
    assert sorted(regen) == sorted(names)
    for n in names:
        recorded = (DATA / "maps" / f"{n}.osu").read_bytes().decode("utf-8")
        assert _norm(regen[n]) == _norm(recorded), n


def _structure(name):
    from osu_sim.beatmap import CIRCLE, SLIDER, SPINNER, build_beatmap
    from osu_sim.osufile import parse_osu
    bm = build_beatmap(parse_osu(DATA / "maps" / f"{name}.osu"))
    kinds = Counter(o.kind for o in bm.objects)
    nested = Counter()
    for o in bm.objects:
        if o.kind == SLIDER:
            for e in o.events:
                nested["tail" if e[0] == 2 else "tick_or_repeat"] += 1
    return dict(heads=kinds[CIRCLE] + kinds[SLIDER], sliders=kinds[SLIDER], spinners=kinds[SPINNER],
                tails=nested["tail"], ticks=nested["tick_or_repeat"])


PERFECT = [c for c in CASES if c["variant"] == "perfect"]


@pytest.mark.parametrize("case", PERFECT, ids=[f"{c['dir']}-{c['map']}" for c in PERFECT])
def test_conversion_matches_native_structure(case):
    rec = json.loads((DATA / case["copy"]).read_text(encoding="utf-8"))
    nat = Counter(rec["real_totals"])
    s = _structure(case["map"])
    assert s["tails"] == s["sliders"]
    assert nat["Great"] + nat["Ok"] + nat["Meh"] + nat["Miss"] == s["heads"]
    assert nat["SliderTailHit"] + nat["IgnoreMiss"] == s["tails"]
    assert nat["LargeTickHit"] + nat["LargeTickMiss"] == s["ticks"]
    assert nat["spin_hit"] + nat["spin_miss"] == s["spinners"]
    if not (nat["Miss"] or nat["LargeTickMiss"] or nat["IgnoreMiss"] or nat["spin_miss"]):
        assert rec["real_max_combo"] == s["heads"] + s["ticks"] + s["tails"] + s["spinners"]


@functools.lru_cache(maxsize=None)
def _trace(name, variant, seed, rate):
    per_step, combo, steps, timeout = paritylib.sim_trace(DATA / "maps" / f"{name}.osu", variant, seed, rate)
    return paritylib.totals(per_step), combo, steps, timeout


@pytest.mark.parametrize("case", CASES, ids=[f"{c['dir']}-{c['map']}-{c['variant']}" for c in CASES])
def test_judgement_parity_against_native_record(case):
    rec = json.loads((DATA / case["copy"]).read_text(encoding="utf-8"))
    assert rec["mismatched_steps"] == 0 and rec["totals_equal"] is True and rec["first_mismatches"] == []
    assert float(rec.get("game_rate", 1.0)) == case["rate"]
    tot, combo, steps, timeout = _trace(case["map"], case["variant"], case["seed"], case["rate"])
    assert not timeout
    assert tot == rec["real_totals"], (tot, rec["real_totals"])
    assert combo == rec["real_max_combo"]
    assert steps == rec["steps_real"]
    assert tot == rec["sim_totals"] and steps == rec["steps_sim"] and combo == rec["sim_max_combo"]


def test_zz_every_loaded_module_from_install():
    """Runs last (file order): after the parity tests imported rules, env, rate, autopilot, generator, pool, the
    loaded osu_sim / osu_rl modules all come from the installation when OSU_SIM_EXPECT_UNDER is set."""
    test_installed_package_location()
