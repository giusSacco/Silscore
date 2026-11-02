"""Silscore model fitting, persistence, and bundled pretrained model."""

from __future__ import annotations

from collections import defaultdict
from importlib import resources
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .scorer import PhysicsAnchoredScorer
from .thermo import TurnerNNBackend

LAM = 1.0
MU = 1.0
MIN_N = 30
ALPHA = 0.05
_INTERCEPT = "intercept"


class PotencyModel(PhysicsAnchoredScorer):
    """Silscore parameterization with an optional unpenalized intercept."""

    def __init__(self, *, intercept: bool = True, intercept_value: float = 1.0,
                 identity_terms=None, fit_floor: bool = False, **kwargs) -> None:
        if fit_floor:
            raise ValueError("PotencyModel does not support fit_floor=True.")
        if identity_terms is None:
            identity_terms = []
        kwargs.setdefault("lam", LAM)
        kwargs.setdefault("mu", MU)
        super().__init__(identity_terms=identity_terms, fit_floor=False, **kwargs)
        if not np.isfinite(intercept_value) or intercept_value == 0:
            raise ValueError("intercept_value must be finite and nonzero.")
        self.intercept = bool(intercept)
        self.intercept_value = float(intercept_value)

    def _extra(self, dg_centered, gate) -> tuple:
        """Build global-energy columns and, when enabled, a constant column."""
        columns = [dg_centered, dg_centered ** 2]
        names = ["c1", "c2"]
        if self.intercept:
            columns.append(np.full(len(dg_centered), self.intercept_value))
            names.append(_INTERCEPT)
        return np.column_stack(columns), names

    def _store(self, weights, names, coefficients, identity_coefficients=None) -> None:
        super()._store(weights, names, coefficients, identity_coefficients)
        values = dict(zip(names, map(float, coefficients)))
        self._parameters[_INTERCEPT] = values.get(_INTERCEPT, 0.0)

    def predict(self, records) -> np.ndarray:
        """Return scores including the fitted intercept, when enabled."""
        prediction = super().predict(records)
        if not self.intercept or not len(prediction):
            return prediction
        # The base predictor restores the label mean; add the fitted constant term.
        return prediction + self._parameters[_INTERCEPT] * self.intercept_value

    def design_matrix(self, records) -> np.ndarray:
        """Return reduced stack, global, and identity columns in fitting order.

        Use the training energy mean if fitted, otherwise the supplied records' mean.
        """
        stacks, dg, gate, identities = self._design(records)
        dg_mean = self._dgbar if self._parameters is not None else float(dg.mean())
        extras, _ = self._extra(dg - dg_mean, gate)
        blocks = [stacks @ self._meanzero_basis(stacks.shape[1]), extras]
        if identities.shape[1]:
            blocks.append(identities)
        return np.column_stack(blocks)

    def report(self) -> dict:
        """Add the effective intercept to the fitted coefficient report."""
        report = super().report()
        report[_INTERCEPT] = (
            self._parameters[_INTERCEPT] * self.intercept_value if self.intercept else 0.0
        )
        return report


def _fit_base(records) -> PotencyModel:
    return PotencyModel(backend=TurnerNNBackend(), identity_terms=[]).fit(records)


def _identity_z_scores(residuals, records) -> tuple[dict, int, float]:
    """Return cell z-scores, tested-cell count, and a two-sided Bonferroni cutoff."""
    cells = defaultdict(list)
    for record, residual in zip(records, np.asarray(residuals, float)):
        for position, base in enumerate(record.guide, start=1):
            cells[position, base].append(residual)
    tested = {key: np.asarray(values) for key, values in cells.items() if len(values) >= MIN_N}
    if not tested:
        raise ValueError(f"No identity cell has at least {MIN_N} guides.")
    threshold = float(norm.ppf(1 - ALPHA / (2 * len(tested))))
    z_scores = {
        key: float(values.mean() / (values.std(ddof=1) / np.sqrt(len(values))))
        for key, values in tested.items()
    }
    return z_scores, len(tested), threshold


def _select_identity_terms(residuals, records) -> tuple[list, dict, int, float]:
    """Select residual identity terms above the cutoff, ordered by absolute z-score."""
    z_scores, count, threshold = _identity_z_scores(residuals, records)
    selected = sorted(
        [key for key, value in z_scores.items() if abs(value) >= threshold],
        key=lambda key: -abs(z_scores[key]),
    )
    return selected, z_scores, count, threshold


def select_identity_terms(records) -> tuple[list, dict, int, float]:
    """Screen identities using residuals from a fit without identity terms.

    Return selected (1-based position, base) pairs, z-scores, tested-cell count,
    and the Bonferroni cutoff, using only the supplied training records.
    """
    base = _fit_base(records)
    efficacy = np.asarray([record.efficacy for record in records], float)
    return _select_identity_terms(efficacy - base.predict(records), records)


class Silscore(PotencyModel):
    """Fixed 19-nt model with training-only identity screening and JSON persistence."""

    def __init__(self, *, identity_terms=()):
        super().__init__(backend=TurnerNNBackend(), identity_terms=identity_terms, lam=LAM, mu=MU)
        self.identity_terms = tuple(tuple(term) for term in self.identity_terms)

    def fit(self, records, y=None):
        """Select identity terms, refit all coefficients, and return self.

        Supply efficacy in the records; separate y values are rejected so that
        screening and fitting use the same labels.
        """
        self._parameters = None
        if y is not None:
            raise ValueError("Supply efficacy in records so screening and fitting use the same labels.")
        records = list(records)
        if not records or any(record.efficacy is None for record in records):
            raise ValueError("Training requires nonempty records with efficacy labels.")
        terms, _, _, _ = select_identity_terms(records)
        self.identity_terms = tuple(tuple(term) for term in terms)
        return super().fit(records)

    def to_dict(self) -> dict:
        """Return JSON-compatible coefficients and centering means for a fitted model."""
        if self._parameters is None:
            raise RuntimeError("Fit the model before saving it.")
        parameters = self._parameters
        return {
            "model": "silscore", "schema": 1, "backend": "xia1998-wc-37C",
            "length": self.length, "lam": self.lam, "mu": self.mu,
            "identity_terms": [[position, base] for position, base in self.identity_terms],
            "w": [float(value) for value in parameters["wvec"]],
            "c1": float(parameters["c1"]), "c2": float(parameters["c2"]),
            "intercept": float(parameters[_INTERCEPT]) * self.intercept_value,
            "theta": [float(value) for value in parameters["idcoef"]],
            "y_mean": self._ybar, "dg_mean": self._dgbar,
        }

    def save(self, path) -> None:
        """Save the fitted model as JSON, creating parent directories if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def from_dict(cls, data):
        """Validate serialized model settings and coefficients, then restore the fit."""
        if (data.get("model"), data.get("schema"), data.get("backend")) != (
            "silscore", 1, "xia1998-wc-37C"
        ):
            raise ValueError("Unsupported model, schema, or thermodynamic parameters.")
        if data["length"] != 19:
            raise ValueError("Silscore expects 19-nt guides.")
        terms = data["identity_terms"]
        if any(not isinstance(position, int) or isinstance(position, bool) for position, _ in terms):
            raise ValueError("Identity positions must be integers.")
        if len({tuple(term) for term in terms}) != len(terms):
            raise ValueError("Duplicate identity terms.")
        if (data["lam"], data["mu"]) != (LAM, MU):
            raise ValueError("Silscore uses the published penalties lambda=mu=1.")
        model = cls(identity_terms=terms)
        weights = np.asarray(data["w"], float)
        identity_coefficients = np.asarray(data["theta"], float)
        scalars = np.asarray([
            data[key] for key in ("lam", "mu", "c1", "c2", "intercept", "y_mean", "dg_mean")
        ], float)
        if weights.shape != (18,) or identity_coefficients.shape != (len(terms),):
            raise ValueError("Incorrect coefficient dimensions.")
        if not all(np.isfinite(values).all() for values in (weights, identity_coefficients, scalars)):
            raise ValueError("Coefficients must be finite.")
        if np.any(scalars[:2] < 0):
            raise ValueError("Penalties must be nonnegative.")
        if not np.isclose(weights.sum(), 0, atol=1e-10, rtol=0):
            raise ValueError("Stacking weights must sum to zero.")
        model._parameters = {
            "wvec": weights, "c1": float(data["c1"]), "c2": float(data["c2"]),
            "gamma": 0.0, "idcoef": identity_coefficients,
            _INTERCEPT: float(data["intercept"]),
        }
        model._ybar = float(data["y_mean"])
        model._dgbar = float(data["dg_mean"])
        return model

    @classmethod
    def load(cls, path):
        """Load a fitted model from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))


def load_default_model() -> Silscore:
    """Load the pretrained model bundled with the installed package."""
    model_text = resources.files("silscore").joinpath("resources/silscore.json").read_text()
    return Silscore.from_dict(json.loads(model_text))
