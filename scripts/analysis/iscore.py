

"""Provide the fitted i-Score comparator used by the canonical analyses."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

BASE_IDX = {"A": 0, "C": 1, "G": 2, "U": 3}
# Selected once on the canonical Huesken training set, then fixed in every re-fit.
ARTICLE_ALPHA = 10.0

def iscore_features(records) -> np.ndarray:

    X = np.zeros((len(records), 19 * 4), dtype=np.float64)
    for i, r in enumerate(records):
        for pos, base in enumerate(r.guide[:19]):
            X[i, pos * 4 + BASE_IDX[base]] = 1.0
    return X

class FittedIScoreScorer:

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._model = None

    def fit(self, records, y=None) -> "FittedIScoreScorer":
        if y is None:
            y = np.array([r.efficacy for r in records], dtype=np.float64)
        self._model = Ridge(alpha=self.alpha).fit(iscore_features(records), y)
        return self

    def predict(self, records) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("FittedIScoreScorer: call fit() before predict()")
        return self._model.predict(iscore_features(records))
