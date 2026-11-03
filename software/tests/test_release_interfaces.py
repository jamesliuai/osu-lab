"""CPU checks for the exported package's portable interfaces, using synthetic inputs only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    import numba
    import torch
    # Limits apply to this test process only; no installation or external process changes.
    torch.set_num_threads(1)
    numba.set_num_threads(1)
    assert torch.get_num_threads() == 1
    assert numba.get_num_threads() == 1
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def generated_pool():
    from osu_sim.beatmap import build_beatmap
    from osu_sim.generator import GenParams, generate
    from osu_sim.pool import compile_pool
    params = GenParams(duration_ms=1000.0, bpm=120.0)
    content = generate(np.random.default_rng(14073), params)
    return compile_pool([("generated_release_interface_test", build_beatmap(content))])


def test_generated_pool_save_load_roundtrip(generated_pool, tmp_path):
    from osu_sim.pool import MapPool
    path = tmp_path / "generated.npz"
    generated_pool.save(path)
    loaded = MapPool.load(path)
    assert loaded.keys == generated_pool.keys
    assert loaded.n_maps == 1
    for table in ("maps", "objs", "nest", "path"):
        expected, actual = getattr(generated_pool, table), getattr(loaded, table)
        assert expected.keys() == actual.keys()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])


def test_s1_policy_cpu_and_packaged_prior_origin():
    import torch
    import osu_sim
    from osu_sim.policy import Agent
    initial_path = tuple(sys.path)
    agent = Agent(cursor_head="s1", hidden=16, proj_dim=16, trunk=8, width=0.25).eval()
    assert tuple(sys.path) == initial_path
    assert next(agent.parameters()).device.type == "cpu"
    assert torch.isfinite(agent.prior_log_masses).all()
    action, value, state = agent.act(
        torch.zeros((1, 3, 192, 256), dtype=torch.uint8),
        torch.zeros((1, 8)), agent.initial_state(1, "cpu"),
        torch.ones((1,), dtype=torch.bool), mode="deterministic")
    assert action.cursor.shape == (1, 2)
    assert torch.isfinite(action.cursor).all() and torch.isfinite(value).all()
    assert state.device.type == "cpu"
    expected_root = Path(os.environ.get("OSU_SIM_EXPECT_UNDER", Path(osu_sim.__file__).resolve().parent.parent)).resolve()
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in ("osu_sim", "osu_rl") and getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(expected_root), (name, module.__file__)
    assert "osu_rl.policy.priors" in sys.modules


@pytest.mark.parametrize("entrypoint", ("environment", "evaluator", "training_preflight"))
def test_missing_pool_is_refused(entrypoint):
    from osu_sim.env import make_env
    from osu_sim.evaluate import SimEvaluator
    from osu_sim.train import configured_pool_paths
    with pytest.raises(KeyError, match="pool"):
        if entrypoint == "environment":
            make_env({}, device="cpu")
        elif entrypoint == "evaluator":
            SimEvaluator({}, device="cpu")
        else:
            configured_pool_paths({"env": {}, "eval": {}})


@pytest.mark.parametrize("match_by", ("path", "sha256"))
def test_explicit_registry_refuses_generated_pool(match_by, generated_pool, tmp_path, monkeypatch):
    import osu_sim.train as train
    pool = tmp_path / "generated.npz"
    generated_pool.save(pool)
    entry = ({"path": str(pool)} if match_by == "path" else
             {"sha256": hashlib.sha256(pool.read_bytes()).hexdigest()})
    registry = tmp_path / "supplied_registry.json"
    registry.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
    monkeypatch.setenv("OSU_SEALED_REGISTRY", str(registry))
    if match_by == "path":
        def forbidden_hash(*args, **kwargs):
            pytest.fail("A path-matched pool must be refused before opening it for hashing")
        monkeypatch.setattr(train, "_sha256_file", forbidden_hash)
    config = {"env": {"pool": str(pool)}, "eval": {"pool": str(pool)}}
    with pytest.raises(SystemExit, match="REFUSED.*registered sealed pool"):
        train.check_sealed_registry(config, root=tmp_path)
