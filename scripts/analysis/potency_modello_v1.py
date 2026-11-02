"""Dataset splits and settings shared by the manuscript analyses."""
from __future__ import annotations
import csv
import os
import sys
from collections import namedtuple
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from silscore.model import ALPHA, LAM, MIN_N, MU, PotencyModel, select_identity_terms
from silscore.records import SiRNARecord
from silscore.thermo import TurnerNNBackend
SPLIT = ROOT / 'data' / 'processed' / 'huesken_canonical_split.csv'
B = TurnerNNBackend()
INTERCEPT_NAME = 'intercept'
EXPECTED_CELLS = [(1, 'U'), (19, 'A'), (18, 'G'), (2, 'C'), (1, 'C'), (14, 'G')]
MODEL_NAME = 'modello principale (V1)'

def _integer_from_env(name: str, default: int, minimum: int=1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f'{name}={raw!r} is not an integer') from e
    if value < minimum:
        raise ValueError(f'{name}={value} must be at least {minimum}')
    return value
SEED = _integer_from_env('POTENCY_SEME', 0, minimum=0)
B_BOOT = _integer_from_env('POTENCY_B_BOOT', 4000, minimum=2)

def fit_model(train, cells=(), *, intercept: bool=True) -> PotencyModel:
    """Fit the shared model with the supplied identity terms, without screening."""
    return PotencyModel(backend=B, identity_terms=list(cells), intercept=intercept).fit(train)
Pool = namedtuple('Pool', 'recs y canonical species rows')
EXPECTED_COLUMNS = ('record_id', 'guide', 'overhang', 'norm_activity', 'split', 'train_all_human', 'test_all_human', 'train_rodent', 'test_rodent')
N_ROWS, N_TRAIN, N_TEST = (2431, 2182, 249)
N_TRAIN_TARGET = 489

def load_rows() -> list[dict]:
    """Read Huesken rows in source order and require the canonical 2182/249 split."""
    if not SPLIT.exists():
        raise FileNotFoundError(f'missing {SPLIT}: provide the canonical CSV; see data/SOURCES.md')
    with SPLIT.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != N_ROWS:
        raise ValueError(f'expected {N_ROWS} rows, found {len(rows)}')
    missing = [c for c in EXPECTED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f'split is missing columns {missing}')
    split_values = {r['split'] for r in rows}
    if split_values != {'train', 'test'}:
        raise ValueError(f'split values {sorted(split_values)}, expected only train/test')
    n_tr = sum((1 for r in rows if r['split'] == 'train'))
    if (n_tr, len(rows) - n_tr) != (N_TRAIN, N_TEST):
        raise ValueError(f'split {n_tr}/{len(rows) - n_tr}, expected {N_TRAIN}/{N_TEST}')
    return rows

def _record(r: dict) -> SiRNARecord:
    return SiRNARecord(dataset='huesken_primary', id=r['record_id'], guide=r['guide'], efficacy=float(r['norm_activity']))

def load_canonical_split() -> tuple[list, list]:
    """Return the canonical training and test records, preserving source order."""
    rows = load_rows()
    tr = [_record(r) for r in rows if r['split'] == 'train']
    te = [_record(r) for r in rows if r['split'] != 'train']
    return (tr, te)

def load_pool() -> Pool:
    """Return aligned records, labels, canonical-train mask, species, and source rows."""
    rows = load_rows()
    recs, y, canonical, species = ([], [], [], [])
    for r in rows:
        recs.append(_record(r))
        y.append(float(r['norm_activity']))
        canonical.append(r['split'] == 'train')
        is_human = r['train_all_human'] == '1' or r['test_all_human'] == '1'
        is_rodent = r['train_rodent'] == '1' or r['test_rodent'] == '1'
        if is_human == is_rodent:
            raise ValueError(f"record {r['record_id']}: human={is_human}, rodent={is_rodent} — the two columns must partition the records")
        species.append('human' if is_human else 'rodent')
    return Pool(recs, np.asarray(y, float), np.asarray(canonical, bool), np.asarray(species), rows)

def fixed_target_split(species: np.ndarray, rng=None) -> tuple[np.ndarray, ...]:
    """Return indices for 489 human, 489 rodent, and 1453 held-out human records.

    Sample human training records without replacement; both training pools
    share the remaining human records as their test set.
    """
    human = np.flatnonzero(species == 'human')
    rodent = np.flatnonzero(species == 'rodent')
    if (len(human), len(rodent)) != (1942, N_TRAIN_TARGET):
        raise ValueError(f'human/rodent partition {len(human)}/{len(rodent)}, expected 1942/{N_TRAIN_TARGET}')
    rng = np.random.default_rng(SEED) if rng is None else rng
    human_train = rng.choice(human, size=N_TRAIN_TARGET, replace=False)
    selected = set(human_train.tolist())
    test = np.asarray(sorted(set(human.tolist()) - selected), dtype=int)
    if len(test) != 1453 or selected.intersection(test.tolist()):
        raise ValueError('human training and fixed test do not form the 489/1453 partition')
    if set(rodent.tolist()).intersection(test.tolist()):
        raise ValueError('rodent training overlaps the human test')
    return (human_train, rodent, test)

def term_label(cell) -> str:
    p, b = cell
    return f'{b}{p}'
RESULTS_DIR = ROOT / 'data' / 'cache'
RUNNER_KEY = 'POTENCY_RUNNER'

def runner_mode() -> str:
    return os.environ.get(RUNNER_KEY, '')

def results_directory() -> Path:
    """Resolve the canonical cache directory or its diagnostic subdirectory."""
    subdirectory = os.environ.get('POTENCY_OUTPUT_SUBDIR', '')
    if subdirectory not in ('', 'diagnostico'):
        raise ValueError("POTENCY_OUTPUT_SUBDIR must be empty or 'diagnostico'")
    return RESULTS_DIR / subdirectory if subdirectory else RESULTS_DIR

def result_path(name: str) -> Path:
    """Create the output directory and restrict canonical writes to the full runner."""
    directory = results_directory()
    if directory == RESULTS_DIR and runner_mode() != 'completo':
        raise RuntimeError(f"{name}: {RESULTS_DIR} contains the manuscript results and can only be written by a full scripts/reproduce.py run. For a manual run, set POTENCY_OUTPUT_SUBDIR=diagnostico")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name
