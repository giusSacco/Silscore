"""Prepare nucleotide-substitution summaries for the fixed-model heatmap."""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from silscore import SiRNARecord, load_default_model  # noqa: E402
from nc_datasets import load_dataset  # noqa: E402

BASES = "UACG"
DATASETS = ("huesken", "mix", "taka")
COUNTS = (249, 472, 702)
INPUTS = (
    ("processed/huesken_canonical_split.csv", "03508de5bb7a66208c7f3910e2d99300356c16fed88b0a93e72f2fff4ea6f074"),
    ("raw/Mix.csv", "10c34deaa5a02e5a168e6951edbd7144fe9025ad40a679c3defb8fe726620c87"),
    ("raw/Taka.csv", "221672544ee0bd7e4c8029d9e8f5d7463c90aa9f5623976002b5e68c35300b2f"),
)


def load_guides(data_dir):
    hashes = []
    for relative_path, expected_hash in INPUTS:
        path = data_dir / relative_path
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_hash:
            raise ValueError(f"{relative_path}: input differs from the canonical source.")
        hashes.append(actual)
    with (data_dir / INPUTS[0][0]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2431 or {row["split"] for row in rows} != {"train", "test"}:
        raise ValueError("Expected the canonical 2,431-row Huesken split.")
    guides = [[row["guide"] for row in rows if row["split"] == "test"]]
    guides.extend([
        [record.guide for record in load_dataset(dataset, data_dir=data_dir / "raw")
         if record.efficacy is not None]
        for dataset in DATASETS[1:]
    ])
    for dataset, sequences, expected in zip(DATASETS, guides, COUNTS):
        if len(sequences) != expected:
            raise ValueError(f"{dataset}: expected {expected} guides, got {len(sequences)}.")
        if any(len(guide) != 19 or set(guide) - set(BASES) for guide in sequences):
            raise ValueError(f"{dataset}: expected uppercase 19-nt RNA guides.")
    return guides, hashes


def summarize(model, guides):
    variants = [
        SiRNARecord("nucleotide_heatmap", str(index), guide[:position] + base + guide[position + 1:])
        for index, guide in enumerate(guides)
        for position in range(19)
        for base in BASES
    ]
    scores = np.asarray(model.predict(variants), dtype=float).reshape(len(guides), 19, 4)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite variant predictions.")
    means = scores.mean(axis=0).T
    column_means = means.mean(axis=0)
    centered = means - column_means
    np.testing.assert_allclose(centered.sum(axis=0), 0, atol=1e-12, rtol=0)
    # Cross-check centering against within-guide contrasts.
    np.testing.assert_allclose(
        centered, (scores - scores.mean(axis=2, keepdims=True)).mean(axis=0).T,
        atol=1e-12, rtol=0,
    )
    return means, column_means, centered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="local data directory with processed/ and raw/ inputs")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "cache" / os.environ.get("POTENCY_OUTPUT_SUBDIR", "")
                        / "nucleotide_heatmap.npz")
    args = parser.parse_args(argv)
    guides, hashes = load_guides(args.data_dir)
    model = load_default_model()
    model_bytes = resources.files("silscore").joinpath("resources/silscore.json").read_bytes()
    results = [summarize(model, sequences) for sequences in guides]
    means, column_means, centered = [np.stack(values) for values in zip(*results)]
    stddev = centered.std(axis=0, ddof=0)
    pairwise = np.sqrt(((centered[0] - centered[1]) ** 2
                        + (centered[0] - centered[2]) ** 2
                        + (centered[1] - centered[2]) ** 2) / 9)
    np.testing.assert_allclose(stddev, pairwise, atol=1e-14, rtol=1e-12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, datasets=np.asarray(DATASETS), bases=np.asarray(list(BASES)),
        positions=np.arange(1, 20), n_guides=np.asarray(COUNTS),
        mean_scores=means, column_means=column_means, centered_scores=centered,
        stddev=stddev, sd_ddof=np.asarray(0),
        model_sha256=np.asarray(hashlib.sha256(model_bytes).hexdigest()),
        input_sha256=np.asarray(hashes),
    )
    print(json.dumps({
        "output": args.output.name, "datasets": list(DATASETS), "n_guides": list(COUNTS),
        "model_refit": False, "mean_stddev": float(stddev.mean()),
        "max_abs_column_sum": float(abs(centered.sum(axis=1)).max()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
