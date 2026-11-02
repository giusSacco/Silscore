"""Record-bootstrap confidence intervals for correlations and paired contrasts.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

import potency_modello_v1 as V1

B_BOOT = V1.B_BOOT
SEED = V1.SEED


def correlation_ci(pred, y, b=B_BOOT):
    """Pearson correlation and its record-bootstrap 95% interval."""
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(b):
        i = rng.integers(0, len(y), len(y))
        if np.std(y[i]) > 0:
            vals.append(pearsonr(pred[i], y[i])[0])
    return float(pearsonr(pred, y)[0]), *np.percentile(vals, [2.5, 97.5])


def paired_correlation_ci(pa, pb, y, b=B_BOOT):
    """Pearson(pa, y) minus Pearson(pb, y), with a paired record-bootstrap 95% interval."""
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(b):
        i = rng.integers(0, len(y), len(y))
        if np.std(y[i]) > 0:
            deltas.append(pearsonr(pa[i], y[i])[0] - pearsonr(pb[i], y[i])[0])
    point = pearsonr(pa, y)[0] - pearsonr(pb, y)[0]
    return float(point), *np.percentile(deltas, [2.5, 97.5])
