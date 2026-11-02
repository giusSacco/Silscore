"""Sequence-only DSIR features and estimator used by the figure producers."""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.linear_model import LassoLarsCV

BASES = ("A", "C", "G", "U")
MOTIFS = ["".join(m) for k in (1, 2, 3)
          for m in itertools.product(BASES, repeat=k)]
MOTIF_INDEX = {motif: i for i, motif in enumerate(MOTIFS)}


def normalize(seq: str) -> str:
    """Normalize DNA spelling to the RNA alphabet."""
    return seq.upper().replace("T", "U")


def guide_with_overhang(guide: str, overhang: str) -> str:
    """Append a two-base siRNA overhang to a 19-base guide."""
    g, o = normalize(guide), normalize(overhang)
    if len(g) != 19:
        raise ValueError(f"guide length {len(g)}, expected 19: {g!r}")
    if len(o) != 2:
        raise ValueError(f"overhang length {len(o)}, expected 2: {o!r}")
    return g + o


def sparse_features(seqs: list[str], length: int) -> np.ndarray:
    X = np.zeros((len(seqs), 4 * length), dtype=np.float64)
    for i, seq in enumerate(seqs):
        if len(seq) < length:
            raise ValueError(f"sequence length {len(seq)}, expected at least {length}: {seq!r}")
        for pos, base in enumerate(seq[:length]):
            X[i, pos * 4 + BASES.index(base)] = 1.0
    return X


def spectrum_features(seqs: list[str], length: int) -> np.ndarray:
    X = np.zeros((len(seqs), len(MOTIFS)), dtype=np.float64)
    for i, seq in enumerate(seqs):
        if len(seq) < length:
            raise ValueError(f"sequence length {len(seq)}, expected at least {length}: {seq!r}")
        for k in (1, 2, 3):
            for pos in range(length - k + 1):
                X[i, MOTIF_INDEX[seq[pos:pos + k]]] += 1.0
    return X


def build_features(seqs: list[str], representation: str) -> np.ndarray:
    if representation.startswith("sparse-"):
        return sparse_features(seqs, int(representation.split("-")[1]))
    if representation.startswith("spectrum-"):
        return spectrum_features(seqs, int(representation.split("-")[1]))
    if representation.startswith("composite-"):
        first, second = representation.split("-")[1].split("/")
        return np.hstack([sparse_features(seqs, int(first)),
                          spectrum_features(seqs, int(second))])
    raise ValueError(f"unknown representation: {representation!r}")


class LarsLassoDSIR:
    """DSIR's column-normalized LARS/Lasso fit."""

    def __init__(self, cv=5, normalize_columns: bool = True,
                 max_iter: int = 2000) -> None:
        self.cv = cv
        self.normalize_columns = normalize_columns
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = None
        self.alpha_ = None
        self.n_nonzero_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LarsLassoDSIR":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mean_x, mean_y = X.mean(axis=0), y.mean()
        centered = X - mean_x
        if self.normalize_columns:
            norms = np.sqrt((centered ** 2).sum(axis=0))
            constant = norms == 0.0
            safe_norms = np.where(constant, 1.0, norms)
        else:
            safe_norms = np.ones(X.shape[1])
            constant = np.zeros(X.shape[1], dtype=bool)
        scaled = centered / safe_norms
        scaled[:, constant] = 0.0

        model = LassoLarsCV(cv=self.cv, fit_intercept=False, max_iter=self.max_iter)
        model.fit(scaled, y - mean_y)
        self.coef_ = model.coef_ / safe_norms
        self.coef_[constant] = 0.0
        self.intercept_ = mean_y - float(self.coef_ @ mean_x)
        self.alpha_ = float(model.alpha_)
        self.n_nonzero_ = int((self.coef_ != 0).sum())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_
