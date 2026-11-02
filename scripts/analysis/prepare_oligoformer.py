"""Recompute the OligoFormer figure cache from saved per-record predictions."""
import argparse
import csv
import sys
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPLIT_SEEDS = range(43, 48)
TRAINING_SEEDS = range(5)
PREDICTION_COLUMNS = [f"split_{s}_seed_{t}" for s in SPLIT_SEEDS for t in TRAINING_SEEDS]

PAIRS = [("main", "eval_huesken249"), ("main", "eval_mix472"), ("main", "eval_taka702"),
         ("human", "eval_human1453"), ("rodent", "eval_human1453")]
B = 4000


def correlations(y, p):
    yc, pc = y-y.mean(axis=1, keepdims=True), p-p.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(yc*yc, axis=1)[:, None]*np.sum(pc*pc, axis=1))
    assert np.all(denominator > 0)
    return np.sum(yc[:, :, None]*pc, axis=1)/denominator


def interval(samples):
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])]


def factorial_weights(selected_splits, selected_seeds):
    a = (selected_splits[:, :, None] == np.arange(5)[None, None, :]).sum(axis=1)/5
    b = (selected_seeds[:, :, None] == np.arange(5)[None, None, :]).sum(axis=1)/5
    return (a[:, :, None]*b[:, None, :]).reshape(-1, 25)


def plot_cache(data):
    """Keep only scalar quantities needed by the two figure notebooks."""
    fields = ("model", "dataset", "n", "oligoformer_r", "ci_low", "ci_high",
              "delta_silscore_minus_oligoformer", "delta_ci_low", "delta_ci_high")
    return dict(source_commit="e2f53ad63387bbe166bf123949151e2bc9bf6ec3",
                protocol=dict(n_splits=5, n_seeds=5, validation_fraction=0.2,
                              bootstrap_replicates=B, point="mean_run_pearson"),
                comparisons=[{k: r[k] for k in fields} for r in data["comparisons"]],
                species_transfer={k: data["species_transfer"][k]
                                  for k in ("rodent_minus_human", "ci_low", "ci_high")})


def load_metrics(path):
    """Read the scalar figure cache and require the 5 × 5 protocol."""
    if not path.is_file():
        raise FileNotFoundError(
            "Missing data/cache/oligoformer.json. Prepare it with: python scripts/analysis/"
            "prepare_oligoformer.py; "
            "verify output/oligoformer/oligoformer.json before copying it to data/cache/.")
    data = json.loads(path.read_text())
    expected = dict(n_splits=5, n_seeds=5, validation_fraction=0.2,
                    bootstrap_replicates=B, point="mean_run_pearson")
    if data.get("protocol") != expected or data.get("source_commit") != "e2f53ad63387bbe166bf123949151e2bc9bf6ec3":
        raise ValueError("Expected complete 5x5 OligoFormer protocol (B=4000)")
    rows = data["comparisons"]
    if len(rows) != 5 or {(r["model"], r["dataset"]) for r in rows} != set(PAIRS):
        raise ValueError("Incomplete OligoFormer comparisons")
    expected_counts = dict(eval_huesken249=249, eval_mix472=472,
                           eval_taka702=702, eval_human1453=1453)
    for row in rows:
        if row.get("n") != expected_counts[row["dataset"]]:
            raise ValueError("Incorrect OligoFormer evaluation count")
        values = [row[k] for k in ("oligoformer_r", "ci_low", "ci_high",
                  "delta_silscore_minus_oligoformer", "delta_ci_low", "delta_ci_high")]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite OligoFormer metrics")
    transfer = data.get("species_transfer", {})
    transfer_keys = ("rodent_minus_human", "ci_low", "ci_high")
    if not all(key in transfer for key in transfer_keys):
        raise ValueError("Incomplete OligoFormer species transfer")
    if not all(math.isfinite(transfer[key]) for key in transfer_keys):
        raise ValueError("Nonfinite OligoFormer species transfer")
    return data


def evaluation_records():
    """Load local evaluation IDs and activities in the original benchmark order."""
    sys.path.insert(0, str(ROOT / "scripts/analysis"))
    import potency_modello_v1 as V1
    from nc_datasets import load_dataset

    pool = V1.load_pool()
    _, _, test = V1.fixed_target_split(pool.species, rng=np.random.default_rng(0))
    return {
        "eval_huesken249": [r for r, train in zip(pool.recs, pool.canonical) if not train],
        "eval_mix472": load_dataset("mix"),
        "eval_taka702": load_dataset("taka"),
        "eval_human1453": [pool.recs[i] for i in test],
    }


def load_predictions(path, expected_ids):
    """Read Silscore and 25 OligoFormer predictions, requiring exact record order."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = ["record_id", "silscore_prediction", *PREDICTION_COLUMNS]
        if reader.fieldnames != fields:
            raise ValueError(f"{path}: unexpected prediction columns")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed prediction row")
    if [r["record_id"] for r in rows] != expected_ids:
        raise ValueError(f"{path}: evaluation IDs or row order differ from the local data")
    matrix = np.array([[float(r[key]) for r in rows] for key in fields[1:]]).T
    if matrix.shape != (len(expected_ids), 26) or not np.isfinite(matrix).all():
        raise ValueError(f"{path}: invalid prediction values")
    if np.any(matrix.std(axis=0) == 0):
        raise ValueError(f"{path}: constant predictions cannot define a correlation")
    return matrix


def main(argv=None):
    """Recompute paired 4000-replicate intervals without training or inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/oligoformer",
                        help="directory containing the five prediction CSVs")
    parser.add_argument("--output", type=Path, default=Path("output/oligoformer"),
                        help="new local output directory")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    datasets = evaluation_records()
    splits = np.random.default_rng(np.random.SeedSequence([0, 1])).integers(0, 5, size=(B, 5))
    seeds = np.random.default_rng(np.random.SeedSequence([0, 2])).integers(0, 5, size=(B, 5))
    weights = factorial_weights(splits, seeds)
    comparisons, means, points = [], {}, {}
    for model, dataset in PAIRS:
        records = datasets[dataset]
        gold = np.array([r.efficacy for r in records], dtype=float)
        path = args.predictions / f"{model}_{dataset.removeprefix('eval_')}.csv"
        matrix = load_predictions(path, [r.id for r in records])
        point = correlations(gold[None, :], matrix[None, :, :])[0]
        sampled = np.empty((B, 26))
        guide_rng = np.random.default_rng(0)
        for start in range(0, B, 32):
            end = min(B, start + 32)
            indices = guide_rng.integers(0, len(gold), size=(end - start, len(gold)))
            sampled[start:end] = correlations(gold[indices], matrix[indices])
        # Cross the independently resampled split and initialization axes.
        mean = np.sum(sampled[:, 1:] * weights, axis=1)
        lo, hi = interval(mean)
        dlo, dhi = interval(sampled[:, 0] - mean)
        comparisons.append(dict(
            model=model, dataset=dataset, n=len(records),
            oligoformer_r=float(point[1:].mean()), ci_low=lo, ci_high=hi,
            delta_silscore_minus_oligoformer=float(point[0] - point[1:].mean()),
            delta_ci_low=dlo, delta_ci_high=dhi,
        ))
        means[model, dataset], points[model, dataset] = mean, point[1:]
    # Human and rodent fits share guide, split, and initialization resamples.
    lo, hi = interval(means["rodent", "eval_human1453"] - means["human", "eval_human1453"])
    transfer = dict(
        rodent_minus_human=float((points["rodent", "eval_human1453"]
                                  - points["human", "eval_human1453"]).mean()),
        ci_low=lo, ci_high=hi,
    )
    result = plot_cache(dict(comparisons=comparisons, species_transfer=transfer))
    args.output.mkdir(parents=True)
    output = args.output / "oligoformer.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
