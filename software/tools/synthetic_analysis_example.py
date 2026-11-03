"""Exercise the portable F1 analysis with artificial counts, never study data.

Usage: python -B tools/synthetic_analysis_example.py --out-root NEW_DIRECTORY

Every input and output is explicitly marked synthetic. No policy, map pool, checkpoint, sealed data or GPU is used.
The tiny input manifest has three artificial records per set, not the study's set sizes. Object counts, outcomes and
accuracy fields are invented solely to check the analysis pipeline; none are a measured result. The example uses
the F1 analysis interface and constants but makes no claim about the active study or trained agents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pd_analysis_f1 as analysis

NOTICE = "SYNTHETIC SOFTWARE DEMONSTRATION ONLY. Artificial counts; no study results or trained-agent evidence."


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as file:
        json.dump(obj, file, sort_keys=True, indent=2, allow_nan=False)
        file.write('\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artificial_record(key, y):
    objects = 100_000
    failures = max(1, min(objects, round(objects * math.exp(y))))
    return {'map': key, 'fc': False, 'max_combo_possible': objects,
            'breaks': failures, 'tail_miss': 0, 'acc': 1.0 - failures / objects}


def build_inputs(root, incomplete=()):
    """Create artificial schema-compatible JSON files only, exclusively, beneath caller's directory."""
    root = Path(root)
    maps = [{'key': f'synthetic:{name}:{i}', 'set': name,
             'band': i if name == 'id' else -1, 'variant': 'plain' if name != 'pattern' else 'burst'}
            for name in analysis.TEST_SETS for i in range(3)]
    manifest_name = 'synthetic_test_manifest.json'
    manifest_sha = write_json(root / manifest_name, {'notice': NOTICE, 'maps': maps})
    runs = []
    for replicate in analysis.REPLICATES:
        for index, size in enumerate(analysis.ARMS):
            name = f'synthetic_n{size}_r{replicate}'
            # Fixed mathematical perturbation gives nonzero residual variance; there is no fitted policy.
            noise = ((replicate * 3 + index * 5) % 7 - 3) * 0.002
            y = -1.3 - 0.1 * replicate - 0.25 * analysis.K_OF[size] + noise
            values = {'id': y, 'tempo': y + 0.2, 'pattern': -0.7 + noise}
            records = [artificial_record(m['key'], values[m['set']]) for m in maps]
            test_name = f'{name}_test.json'
            test_sha = write_json(root / test_name, {'notice': NOTICE, 'detail': {'deterministic': {'maps': records}}})
            probe_keys = [f'synthetic:probe:{replicate}:{size}:{i}' for i in range(3)]
            probe_manifest = f'{name}_probe_manifest.json'
            write_json(root / probe_manifest, {'notice': NOTICE, 'maps': [{'key': key} for key in probe_keys]})
            probe_name = f'{name}_probe.json'
            probe_sha = write_json(root / probe_name, {'notice': NOTICE, 'detail': {'deterministic': {
                'maps': [artificial_record(key, -3.0) for key in probe_keys]}}})
            runs.append({'name': name, 'N': size, 'replicate': replicate,
                         'complete': (size, replicate) not in set(incomplete),
                         'complete_detail': 'synthetic example state', 'eval_jsonl': None,
                         'test_eval': test_name, 'test_eval_sha256': test_sha,
                         'probe_manifest': probe_manifest, 'probe_eval': probe_name, 'probe_eval_sha256': probe_sha})
    batch = root / 'synthetic_batch_summary.json'
    write_json(batch, {'notice': NOTICE, 'schema': analysis.SCHEMA_IN, 'D': 0,
                       'test_manifest': {'path': manifest_name, 'sha256': manifest_sha}, 'runs': runs, 'references': []})
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out-root', required=True, help='new directory; must not already exist')
    args = parser.parse_args(argv)
    root = Path(args.out_root).resolve()
    if root.exists():
        parser.error('--out-root must be a new directory')
    root.mkdir(parents=True, exist_ok=False)
    batch = build_inputs(root)
    result = analysis.clean(analysis.analyze(batch, root=root))
    rendered = analysis.render_text(result)
    violations = analysis.wording_violations(result, rendered)
    if violations:
        raise RuntimeError(f'synthetic output failed wording checks: {violations[:3]}')
    write_json(root / 'SYNTHETIC_analysis.json', {'kind': 'synthetic_software_example', 'notice': NOTICE, 'analysis': result})
    with (root / 'SYNTHETIC_analysis.txt').open('x', encoding='utf-8', newline='\n') as file:
        file.write(NOTICE + '\n\n' + rendered)
    with (root / 'SYNTHETIC_NOTICE.txt').open('x', encoding='utf-8', newline='\n') as file:
        file.write(NOTICE + '\nNo model was trained or evaluated. No map data was read.\n')
    print(NOTICE)
    print(f'Created the artificial input bundle and labelled demonstration outputs in {root}')


if __name__ == '__main__':
    main()
