"""Constrained ridge fitting primitives for Silscore."""

from __future__ import annotations

import warnings

import numpy as np

from .records import SiRNARecord, check_rna
from .thermo import TurnerNNBackend


class PhysicsAnchoredScorer:
    """Fit zero-sum stack weights and global-energy and identity coefficients.

    lam penalizes stack weights; mu penalizes identity coefficients.
    Identity terms use 1-based guide positions.
    """

    def __init__(self, backend: TurnerNNBackend | None = None, lam: float = 1.0,
                 fit_floor: bool = False, identity_terms=(), mu: float = 1.0,
                 length: int = 19) -> None:
        if fit_floor or length != 19:
            raise ValueError("This model supports the constrained 19-nt fit only.")
        if not np.isfinite([lam, mu]).all() or lam < 0 or mu < 0:
            raise ValueError("Penalties must be finite and nonnegative.")
        self._backend = backend or TurnerNNBackend()
        self.lam = lam
        self.fit_floor = fit_floor
        if isinstance(identity_terms, str):
            raise ValueError("identity_terms must be a sequence of (position, base) pairs.")
        self.identity_terms = list(identity_terms)
        self.mu = mu
        self.length = length
        for position, base in self.identity_terms:
            if base not in {"A", "C", "G", "U"} or not 1 <= position <= length:
                raise ValueError(f"Invalid identity term ({position}, {base!r}).")
        self._parameters: dict | None = None
        self._ybar = 0.0
        self._dgbar = 0.0

    def _design(self, records: list[SiRNARecord]) -> tuple:
        """Build stack energies, duplex energies, the 5′ AU flag, and identities."""
        guides = []
        for record in records:
            guide = check_rna(record.guide)
            if len(guide) != self.length:
                raise ValueError(f"Silscore expects {self.length}-nt guides; {record.id} has {len(guide)} nt.")
            guides.append(guide)
        stacks = np.asarray([self._backend.stack_profile(g) for g in guides], dtype=np.float64)
        dg = np.asarray([self._backend.duplex_dg(g) for g in guides], dtype=np.float64)
        gate = np.asarray([1.0 if g[0] in "AU" else 0.0 for g in guides], dtype=np.float64)
        identities = np.zeros((len(guides), len(self.identity_terms)), dtype=np.float64)
        for row, guide in enumerate(guides):
            for column, (position, base) in enumerate(self.identity_terms):
                if guide[position - 1] == base:
                    identities[row, column] = 1.0
        return stacks, dg, gate, identities

    @staticmethod
    def _predict_centered(parameters, stacks, dg_centered, gate, identities):
        output = (stacks @ parameters["wvec"] + parameters["c1"] * dg_centered
                  + parameters["c2"] * dg_centered ** 2 + parameters["gamma"] * gate)
        if identities.shape[1]:
            output = output + identities @ parameters["idcoef"]
        return output

    def fit(self, records: list[SiRNARecord], y: np.ndarray | None = None):
        """Fit labeled guide records, or explicit y values, and return self.

        Rank deficiency emits a RuntimeWarning while retaining the fitted model.
        """
        self._parameters = None
        if not records:
            raise ValueError("Training records must not be empty.")
        if y is None:
            if any(record.efficacy is None for record in records):
                raise ValueError("Every training record must have an efficacy value.")
            y = np.asarray([record.efficacy for record in records], dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (len(records),):
            raise ValueError("Expected one efficacy label per record.")
        if not np.isfinite(y).all():
            raise ValueError("Efficacy labels must be finite.")
        stacks, dg, gate, identities = self._design(records)
        # Reuse these training means when centering new inputs and restoring scores.
        self._ybar = float(y.mean())
        self._dgbar = float(dg.mean())
        self._fit_signed(stacks, dg - self._dgbar, gate, identities, y - self._ybar,
                         stacks.shape[1])
        return self

    def _extra(self, dg_centered, gate) -> tuple:
        """Return unpenalized global-energy columns and their coefficient names."""
        return np.column_stack([dg_centered, dg_centered ** 2]), ["c1", "c2"]

    def _store(self, weights, names, coefficients, identity_coefficients=None) -> None:
        values = dict(zip(names, map(float, coefficients)))
        if identity_coefficients is None:
            identity_coefficients = np.zeros(0)
        self._parameters = {
            "wvec": weights, "c1": values["c1"], "c2": values["c2"],
            "gamma": values.get("gamma", 0.0),
            "idcoef": np.asarray(identity_coefficients, dtype=float),
        }

    @staticmethod
    def _meanzero_basis(size: int) -> np.ndarray:
        """Return an orthonormal basis for vectors whose entries sum to zero."""
        projector = np.eye(size) - np.ones((size, size)) / size
        return np.linalg.svd(projector)[0][:, :size - 1]

    def _fit_signed(self, stacks, dg_centered, gate, identities, y_centered, size) -> None:
        """Solve the constrained ridge problem and store the fitted coefficients."""
        extras, names = self._extra(dg_centered, gate)
        n_extra = extras.shape[1]
        n_identity = identities.shape[1]
        basis = self._meanzero_basis(size)
        # Reduced coordinates enforce the zero-sum constraint on stack weights.
        design = np.column_stack([stacks @ basis, extras, identities])
        rows, targets = [design], [y_centered]
        # Extra least-squares rows apply ridge penalties; global terms stay unpenalized.
        profile_penalty = np.zeros((size - 1, design.shape[1]))
        profile_penalty[:, :size - 1] = np.sqrt(self.lam) * np.eye(size - 1)
        rows.append(profile_penalty)
        targets.append(np.zeros(size - 1))
        if n_identity:
            identity_penalty = np.zeros((n_identity, design.shape[1]))
            identity_penalty[:, size - 1 + n_extra:] = np.sqrt(self.mu) * np.eye(n_identity)
            rows.append(identity_penalty)
            targets.append(np.zeros(n_identity))
        theta, _, rank, _ = np.linalg.lstsq(np.vstack(rows), np.concatenate(targets), rcond=None)
        if rank < design.shape[1]:
            warnings.warn(
                f"Penalized fit is not unique (rank {rank}/{design.shape[1]}); "
                "using the solver's minimum-norm solution.",
                RuntimeWarning, stacklevel=3,
            )
        elif np.linalg.matrix_rank(design) < design.shape[1]:
            warnings.warn(
                "Design columns are dependent; penalties resolve the dependency. "
                "Continuing with the regularized fit.",
                RuntimeWarning, stacklevel=3,
            )
        profile_end = size - 1
        extra_end = profile_end + n_extra
        # Map the reduced positional coefficients back to the 18 stack weights.
        self._store(basis @ theta[:profile_end], names, theta[profile_end:extra_end],
                    theta[extra_end:] if n_identity else None)

    def predict(self, records: list[SiRNARecord]) -> np.ndarray:
        """Score guide records in input order using the stored training means."""
        if self._parameters is None:
            raise RuntimeError("Call fit() before predict().")
        if not records:
            return np.zeros(0)
        stacks, dg, gate, identities = self._design(records)
        return self._predict_centered(
            self._parameters, stacks, dg - self._dgbar, gate, identities
        ) + self._ybar

    def report(self) -> dict:
        """Return fitted coefficients, training means, and the quadratic stationary point."""
        if self._parameters is None:
            raise RuntimeError("Call fit() before report().")
        parameters = self._parameters
        c2 = float(parameters["c2"])
        optimum = self._dgbar - parameters["c1"] / (2.0 * c2) if abs(c2) > 1e-9 else None
        weights = np.asarray(parameters["wvec"], dtype=float)
        identities = {
            f"{base}@{position}": float(coefficient)
            for (position, base), coefficient in zip(self.identity_terms, parameters["idcoef"])
        }
        return {
            "fit_floor": self.fit_floor, "lam": float(self.lam), "mu": float(self.mu),
            "c1": float(parameters["c1"]), "c2": c2, "kappa": -c2,
            "dG_star": optimum, "gamma": float(parameters["gamma"]),
            "dG_bar": self._dgbar, "y_bar": self._ybar, "wvec": list(weights),
            "w_bar": float(weights.mean()), "identity": identities,
        }
