"""CPU-only release-tool checks using pure sequences and newly created artificial input bundles.

Run from the exported package: python -B -m pytest tools/test_pd_tools_synthetic.py -p no:cacheprovider
These checks never read historical pools, study results, sealed files, or checkpoints.
"""
import json
import math

import numpy as np
import pytest

import build_pd_pools as pools
import pd_analysis_f1 as analysis
from synthetic_analysis_example import build_inputs


def test_ols_matches_independent_design_matrix():
    groups = [r for r in analysis.REPLICATES for _ in analysis.ARMS]
    x = np.array([analysis.K_OF[n] for _ in analysis.REPLICATES for n in analysis.ARMS])
    intercepts = np.array([[float(g == r) for r in analysis.REPLICATES] for g in groups])
    design = np.column_stack([intercepts, x])
    y = intercepts @ np.array([-1.0, -1.3, -1.5]) - 0.25 * x + np.array([0.01, -0.02, 0.01] * 3)
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    df = len(y) - design.shape[1]
    variance = np.sum((y - design @ coef) ** 2) / df
    covariance = variance * np.linalg.inv(design.T @ design)
    fit = analysis.ols_fe(y, x, groups)
    assert fit['b'] == pytest.approx(coef[-1])
    assert fit['se'] == pytest.approx(np.sqrt(covariance[-1, -1]))
    assert fit['df'] == df == 5


def test_synthetic_end_to_end_and_f1_design(tmp_path):
    batch = build_inputs(tmp_path)
    result, text = analysis.run_analysis(batch, tmp_path / 'synthetic.json', tmp_path / 'synthetic.txt', root=tmp_path)
    assert result['primary']['status'] == 'Tested'
    assert result['primary']['b'] == pytest.approx(-0.25, abs=0.01)
    assert result['primary']['df'] == 5 and result['primary']['n'] == 9
    assert result['settings']['arms'] == [16, 256, 4096]
    assert result['settings']['replicates'] == [1, 2, 3]
    assert result['settings']['holm_m'] == 6
    assert len(result['power']) == 6
    assert all(row['sxx'] == pytest.approx(32 / 3) for row in result['power'])
    assert all(row['df'] == 5 for row in result['power'])
    assert not analysis.wording_violations(result, text)


def test_incomplete_replicate_suppresses_inference(tmp_path):
    batch = build_inputs(tmp_path, incomplete={(256, 2)})
    result = analysis.analyze(batch, root=tmp_path)
    assert result['complete_replicates'] == [1, 3]
    assert result['primary']['status'] == analysis.NOT_TESTED
    assert result['primary']['b'] is None and result['primary']['p_one_sided'] is None
    assert all(row['status'] == analysis.NOT_TESTED for row in result['secondary'])


def test_hash_mismatch_refuses_before_output(tmp_path):
    batch = build_inputs(tmp_path)
    data = json.loads(batch.read_text(encoding='utf-8'))
    evaluation = tmp_path / data['runs'][0]['test_eval']
    evaluation.write_text(evaluation.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    out_json, out_text = tmp_path / 'synthetic.json', tmp_path / 'synthetic.txt'
    with pytest.raises(SystemExit, match='sha256'):
        analysis.run_analysis(batch, out_json, out_text, root=tmp_path)
    assert not out_json.exists() and not out_text.exists()


def test_reordered_records_refuse():
    with pytest.raises(SystemExit, match='different order'):
        analysis.check_keys([{'map': 'b'}, {'map': 'a'}], ['a', 'b'], 'synthetic', 'synthetic manifest')


def test_zero_failure_and_holm_boundaries():
    assert analysis.y_of(0, 100) == pytest.approx(math.log(0.5 / 100))
    assert analysis.y_of(5, 0) is None
    assert analysis.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_quota_prefixes_and_level_bounds():
    bands, variants, levels = pools.practice_sequence(pools.N_SEQ, [20261101, 1])
    for sequence, shares in ((bands, pools.BAND_SHARES), (variants, pools.VARIANT_SHARES)):
        counts = np.cumsum(np.eye(len(shares))[sequence], axis=0)
        expected = np.arange(1, pools.N_SEQ + 1)[:, None] * np.array(shares)
        assert np.abs(counts - expected).max() <= 1.0 + 1e-9
    for band, (lo, hi) in enumerate(pools.BAND_LEVELS):
        selected = levels[bands == band]
        assert np.all(selected > lo) and np.all(selected < hi)


def test_portable_cli_requires_explicit_new_output_root(tmp_path):
    with pytest.raises(SystemExit):
        pools.main(['monitor'])
    with pytest.raises(SystemExit):
        pools.main(['sealed', '--out-root', str(tmp_path / 'unused')])
    with pytest.raises(SystemExit):
        pools.main(['monitor', '--out-root', str(tmp_path)])
    assert not (tmp_path / 'unused').exists()


def test_analysis_cli_requires_explicit_input_root():
    with pytest.raises(SystemExit):
        analysis.main(['--batch', 'synthetic_batch.json', '--out-json', 'synthetic.json', '--out-txt', 'synthetic.txt'])
