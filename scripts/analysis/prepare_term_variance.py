"""Prepare variance and covariance summaries of the fixed-model score components."""
from __future__ import annotations

import argparse
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
from silscore.thermo import TurnerNNBackend  # noqa: E402
from prepare_nucleotide_heatmap import COUNTS, DATASETS, load_guides  # noqa: E402

COMPONENTS = ("local_stability", "global_stability", "nucleotide_identity")


def contributions(model, guides):
    """Use the saved coefficients without fitting or selecting new terms."""
    state = model.to_dict()
    backend = TurnerNNBackend()
    stacks = np.asarray([backend.stack_profile(guide) for guide in guides], float)
    dg = np.asarray([backend.duplex_dg(guide) for guide in guides], float)
    dg_centered = dg - state["dg_mean"]
    identities = np.asarray([
        [float(guide[position - 1] == base) for position, base in state["identity_terms"]]
        for guide in guides
    ], float).reshape(len(guides), len(state["identity_terms"]))
    terms = np.column_stack([
        stacks @ np.asarray(state["w"]),
        state["c1"] * dg_centered + state["c2"] * dg_centered ** 2,
        identities @ np.asarray(state["theta"]),
    ])
    offset = state["y_mean"] + state["intercept"]
    records = [SiRNARecord("term_variance", str(index), guide)
               for index, guide in enumerate(guides)]
    prediction = np.asarray(model.predict(records), float)
    reconstructed = offset + terms.sum(axis=1)
    if not np.isfinite(terms).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite score contributions.")
    np.testing.assert_allclose(reconstructed, prediction, atol=1e-12, rtol=0)
    return terms, prediction, float(offset), float(np.max(abs(reconstructed - prediction)))


def summarize(terms, prediction):
    means = terms.mean(axis=0)
    centered = terms - means
    covariance = centered.T @ centered / len(terms)
    variances = np.var(terms, axis=0, ddof=0)
    np.testing.assert_allclose(np.diag(covariance), variances, atol=1e-14, rtol=1e-12)
    variance_sum = float(variances.sum())
    if variance_sum <= 0:
        raise ValueError("Variance shares require nonconstant score components.")
    shares = variances / variance_sum
    covariance_total = float(2 * np.triu(covariance, k=1).sum())
    score_variance = float(np.var(prediction, ddof=0))
    np.testing.assert_allclose(score_variance, variance_sum + covariance_total,
                               atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(centered.sum(axis=1), prediction - prediction.mean(),
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(shares.sum(), 1, atol=1e-14, rtol=0)
    return means, variances, shares, covariance, score_variance, variance_sum, covariance_total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="local data directory with processed/ and raw/ inputs")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "cache" / os.environ.get("POTENCY_OUTPUT_SUBDIR", "")
                        / "term_variance.npz")
    args = parser.parse_args(argv)
    guides, hashes = load_guides(args.data_dir)
    model = load_default_model()
    decomposed = [contributions(model, sequences) for sequences in guides]
    summaries = [summarize(terms, scores) for terms, scores, _, _ in decomposed]
    means, variances, shares, covariance, score_variance, variance_sum, covariance_total = (
        list(values) for values in zip(*summaries)
    )
    model_bytes = resources.files("silscore").joinpath("resources/silscore.json").read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, datasets=np.asarray(DATASETS), components=np.asarray(COMPONENTS),
        n_guides=np.asarray(COUNTS), means=np.asarray(means), variances=np.asarray(variances),
        variance_shares=np.asarray(shares), covariance=np.asarray(covariance),
        score_variance=np.asarray(score_variance), sum_term_variances=np.asarray(variance_sum),
        covariance_total=np.asarray(covariance_total),
        variance_ddof=np.asarray(0), model_sha256=np.asarray(hashlib.sha256(model_bytes).hexdigest()),
        input_sha256=np.asarray(hashes), score_offset=np.asarray(decomposed[0][2]),
        reconstruction_max_abs=np.asarray([result[3] for result in decomposed]),
    )
    print(json.dumps({
        "output": args.output.name, "datasets": list(DATASETS), "n_guides": list(COUNTS),
        "components": list(COMPONENTS), "model_refit": False, "variance_ddof": 0,
        "variance_shares": np.asarray(shares).tolist(), "variances": np.asarray(variances).tolist(),
        "score_variance": score_variance, "sum_term_variances": variance_sum,
        "covariance_total": covariance_total, "covariance": np.asarray(covariance).tolist(),
        "reconstruction_max_abs": [result[3] for result in decomposed],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
