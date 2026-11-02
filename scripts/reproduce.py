"""Regenerate the numerical caches for the six figure notebooks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEPS = (
    ('prepare_performance.py', 'potency_vs_dsir.csv', 0),
    ('prepare_target_transfer.py', 'potency_specie_per_figura.csv', 0),
    ('prepare_global_stability.py', 'global_stability.npz', 0),
    ('prepare_weight_profiles.py', 'potency_weight_profiles_v1.csv', 20260821),
    ('prepare_nucleotide_heatmap.py', 'nucleotide_heatmap.npz', None),
    ('prepare_term_variance.py', 'term_variance.npz', None),
)
PERFORMANCE_COLUMNS = {'confronto', 'modello', 'test', 'n', 'valore', 'ic_basso', 'ic_alto', 'esito'}
PROFILE_COLUMNS = {'dataset', 'n', 'model', 'scale', 'coordinate', 'index', 'estimate',
                   'bootstrap_median', 'ci_low', 'ci_high', 'B'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path, bootstrap):
    """Check a generated cache's schema, values, and expected analysis settings."""
    if path.name == 'term_variance.npz':
        with np.load(path, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays['datasets'], ['huesken', 'mix', 'taka'])
            np.testing.assert_array_equal(arrays['components'],
                                          ['local_stability', 'global_stability', 'nucleotide_identity'])
            np.testing.assert_array_equal(arrays['n_guides'], [249, 472, 702])
            for key, shape in [('means', (3, 3)), ('variances', (3, 3)),
                               ('variance_shares', (3, 3)), ('covariance', (3, 3, 3)),
                               ('score_variance', (3,)), ('sum_term_variances', (3,)),
                               ('covariance_total', (3,)), ('reconstruction_max_abs', (3,)),
                               ('score_offset', ())]:
                if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                    raise ValueError(f'{path.name}: invalid {key}')
            if arrays['variance_ddof'].shape != () or arrays['variance_ddof'].item() != 0:
                raise ValueError(f'{path.name}: expected ddof=0')
            if np.any(arrays['variances'] < 0) or np.any(arrays['sum_term_variances'] <= 0):
                raise ValueError(f'{path.name}: invalid variances')
            np.testing.assert_allclose(arrays['covariance'], arrays['covariance'].swapaxes(1, 2), atol=1e-14, rtol=0)
            np.testing.assert_allclose(arrays['variances'], np.diagonal(arrays['covariance'], axis1=1, axis2=2), atol=1e-14, rtol=1e-12)
            np.testing.assert_allclose(arrays['sum_term_variances'], arrays['variances'].sum(axis=1), atol=1e-14, rtol=1e-12)
            np.testing.assert_allclose(arrays['variance_shares'], arrays['variances'] / arrays['sum_term_variances'][:, None], atol=1e-14, rtol=1e-12)
            np.testing.assert_allclose(arrays['variance_shares'].sum(axis=1), 1, atol=1e-14, rtol=0)
            np.testing.assert_allclose(arrays['covariance_total'], arrays['covariance'].sum(axis=(1, 2)) - arrays['sum_term_variances'], atol=1e-14, rtol=1e-12)
            np.testing.assert_allclose(arrays['score_variance'], arrays['sum_term_variances'] + arrays['covariance_total'], atol=1e-14, rtol=1e-12)
            if np.max(arrays['reconstruction_max_abs']) > 1e-12:
                raise ValueError(f'{path.name}: component reconstruction failed')
            for key, shape in [('model_sha256', ()), ('input_sha256', (3,))]:
                if arrays[key].shape != shape or any(len(value) != 64 for value in arrays[key].reshape(-1)):
                    raise ValueError(f'{path.name}: invalid {key}')
        return
    if path.name == 'nucleotide_heatmap.npz':
        with np.load(path, allow_pickle=False) as arrays:
            if list(arrays['datasets']) != ['huesken', 'mix', 'taka'] or ''.join(arrays['bases']) != 'UACG':
                raise ValueError(f'{path.name}: incorrect dataset or base order')
            np.testing.assert_array_equal(arrays['n_guides'], [249, 472, 702])
            np.testing.assert_array_equal(arrays['positions'], np.arange(1, 20))
            for key, shape in [('mean_scores', (3, 4, 19)), ('column_means', (3, 19)),
                               ('centered_scores', (3, 4, 19)), ('stddev', (4, 19))]:
                if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                    raise ValueError(f'{path.name}: invalid {key}')
            if arrays['sd_ddof'].item() != 0 or (arrays['stddev'] < 0).any():
                raise ValueError(f'{path.name}: invalid descriptive standard deviation')
            np.testing.assert_allclose(arrays['centered_scores'],
                                       arrays['mean_scores'] - arrays['column_means'][:, None, :],
                                       atol=1e-12, rtol=0)
            np.testing.assert_allclose(arrays['centered_scores'].sum(axis=1), 0, atol=1e-12, rtol=0)
            np.testing.assert_allclose(arrays['stddev'], arrays['centered_scores'].std(axis=0, ddof=0),
                                       atol=1e-14, rtol=1e-12)
            for key, shape in [('model_sha256', ()), ('input_sha256', (3,))]:
                if arrays[key].shape != shape or any(len(value) != 64 for value in arrays[key].reshape(-1)):
                    raise ValueError(f'{path.name}: invalid {key}')
        return
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as arrays:
            if not arrays.files or any(not np.isfinite(arrays[k]).all() for k in arrays.files):
                raise ValueError(f'{path.name}: empty or nonfinite arrays')
            if arrays['B_BOOT'].item() != bootstrap or arrays['SEME'].item() != 0:
                raise ValueError(f'{path.name}: incorrect bootstrap settings')
        return
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        expected = PROFILE_COLUMNS if path.name == 'potency_weight_profiles_v1.csv' else PERFORMANCE_COLUMNS
        if not rows or set(reader.fieldnames or []) != expected:
            raise ValueError(f'{path.name}: empty table or incorrect columns')
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f'{path.name}: malformed row')
        for value in row.values():
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                raise ValueError(f'{path.name}: nonfinite value')
        if 'B' in row and int(row['B']) != bootstrap:
            raise ValueError(f'{path.name}: incorrect bootstrap count')


def main(argv=None):
    """Run the six cache producers with fixed seeds and record their checksums.

    --quick uses 20 bootstrap samples in a separate diagnostic directory;
    canonical runs use 4000 and require --overwrite when caches already exist.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='20 bootstrap samples, isolated diagnostic output')
    parser.add_argument('--overwrite', action='store_true', help='replace existing canonical figure caches')
    args = parser.parse_args(argv)
    bootstrap = 20 if args.quick else 4000
    output = ROOT / 'data/cache'
    if args.quick:
        output /= 'diagnostico'
    inputs = [ROOT / 'data/processed/huesken_canonical_split.csv', ROOT / 'data/raw/Mix.csv', ROOT / 'data/raw/Taka.csv']
    missing = [str(path.relative_to(ROOT)) for path in inputs if not path.is_file()]
    if missing:
        parser.error('Missing local inputs: ' + ', '.join(missing))
    if not args.quick and not args.overwrite and any((output / name).exists() for _, name, _ in STEPS):
        parser.error('Canonical caches already exist; pass --overwrite to regenerate them')

    # Use the runner's settings rather than inherited analysis overrides.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('POTENCY_') and key != 'PYTHONOPTIMIZE'}
    env.update(POTENCY_SEME='0', POTENCY_B_BOOT=str(bootstrap),
               POTENCY_OUTPUT_SUBDIR='diagnostico' if args.quick else '',
               POTENCY_RUNNER='diagnostico' if args.quick else 'completo')
    code = (sorted((ROOT / 'scripts').glob('*.py')) + sorted((ROOT / 'scripts/analysis').glob('*.py'))
            + sorted((ROOT / 'src/silscore').glob('*.py')))
    manifest = {
        'scope': 'ten figures from six notebooks',
        'mode': 'diagnostic' if args.quick else 'canonical',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'scipy', 'scikit-learn', 'seqfold', 'setuptools')},
        'input_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        'code_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in code},
        'model_sha256': {'src/silscore/resources/silscore.json': sha256(ROOT / 'src/silscore/resources/silscore.json')},
        'steps': [], 'status': 'running',
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    # Remove stale outputs so a failed producer cannot leave an old cache in place.
    for _, name, _ in STEPS:
        (output / name).unlink(missing_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'log').mkdir(exist_ok=True)
    try:
        for script, filename, seed in STEPS:
            step_bootstrap = bootstrap if seed is not None else None
            print(f'{script}: B={step_bootstrap}, seed={seed}', flush=True)
            start = time.monotonic()
            with (output / 'log' / script.replace('.py', '.log')).open('w') as log:
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/analysis' / script)],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{script}: exit {result.returncode}; see its local log')
            validate(output / filename, bootstrap)
            manifest['steps'].append({'script': script, 'bootstrap': step_bootstrap, 'seed': seed,
                                      'output': filename, 'sha256': sha256(output / filename),
                                      'seconds': round(time.monotonic() - start, 2)})
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        manifest['status'] = 'ok'
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = type(error).__name__
        print(str(error), file=sys.stderr)
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if manifest['status'] != 'ok':
        return 1
    print(f'Completed six producers. Outputs: {output.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
