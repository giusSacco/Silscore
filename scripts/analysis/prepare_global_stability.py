"""Prepare curves, binned residuals and counts for the global-stability figure."""
# %%
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import potency_modello_v1 as V1  # noqa: E402
from nc_datasets import load_dataset  # noqa: E402

HIST_EDGES = np.arange(-50.5, -18.49, 2.0)


def fit_reduced(records: list, cells: list, form: str) -> tuple[np.ndarray, ...]:
    """Fit the linear or constant global-stability variant."""
    y = np.asarray([r.efficacy for r in records], float)
    probe = V1.PotencyModel(backend=V1.B, identity_terms=list(cells))
    G, d_g, _gate, identity = probe._design(records)
    y_bar, dg_bar = float(y.mean()), float(d_g.mean())
    d_gc = d_g - dg_bar
    profile = G @ probe._meanzero_basis(G.shape[1])
    blocks = [profile]
    if form == "linear":
        blocks.append(d_gc[:, None])
    elif form != "none":
        raise ValueError(form)
    blocks.append(np.ones((len(y), 1)))
    if identity.shape[1]:
        blocks.append(identity)
    design = np.column_stack(blocks)
    rows, targets = [design], [y - y_bar]
    ridge_profile = np.zeros((profile.shape[1], design.shape[1]))
    ridge_profile[:, :profile.shape[1]] = np.sqrt(V1.LAM) * np.eye(profile.shape[1])
    rows.append(ridge_profile); targets.append(np.zeros(profile.shape[1]))
    if identity.shape[1]:
        ridge_identity = np.zeros((identity.shape[1], design.shape[1]))
        ridge_identity[:, -identity.shape[1]:] = np.sqrt(V1.MU) * np.eye(identity.shape[1])
        rows.append(ridge_identity); targets.append(np.zeros(identity.shape[1]))
    theta = np.linalg.lstsq(np.vstack(rows), np.concatenate(targets), rcond=None)[0]
    cursor = profile.shape[1]
    profile_contribution = profile @ theta[:cursor]
    c1 = float(theta[cursor]) if form == "linear" else 0.0
    cursor += form == "linear"
    intercept = float(theta[cursor]); cursor += 1
    identity_contribution = identity @ theta[cursor:] if identity.shape[1] else np.zeros_like(d_g)
    others = y_bar + profile_contribution + intercept + identity_contribution
    sd_y = float(y.std(ddof=1))
    return np.asarray(d_g), (y - others) / sd_y, c1, dg_bar, sd_y


def components(records: list, fit: V1.PotencyModel) -> tuple[np.ndarray, ...]:
    y = np.asarray([r.efficacy for r in records], float)
    G, d_g, _gate, identity = fit._design(records)
    p = fit._parameters
    d_gc = d_g - fit._dgbar
    others = fit._ybar + G @ p["wvec"] + p[V1.INTERCEPT_NAME]
    if len(p["idcoef"]):
        others += identity @ p["idcoef"]
    vertex = float(fit.report()["dG_star"])
    x_vertex = vertex - fit._dgbar
    maximum = p["c1"] * x_vertex + p["c2"] * x_vertex ** 2
    sd_y = float(y.std(ddof=1))
    residual = (y - others - maximum) / sd_y
    return np.asarray(d_g), residual, (float(d_g.min()), float(d_g.max())), vertex, sd_y


def curve(fit: V1.PotencyModel, x: np.ndarray, sd_y: float) -> np.ndarray:
    p = fit._parameters
    vertex = float(fit.report()["dG_star"])
    xc = x - fit._dgbar
    xv = vertex - fit._dgbar
    return (p["c1"] * xc + p["c2"] * xc ** 2
            - (p["c1"] * xv + p["c2"] * xv ** 2)) / sd_y


def fixed_bins(d_g: np.ndarray, residual: np.ndarray, rng: np.random.Generator) -> list[dict]:
    rows = []
    for i, (low, high) in enumerate(zip(HIST_EDGES[:-1], HIST_EDGES[1:])):
        mask = (d_g >= low) & (d_g <= high if i == len(HIST_EDGES) - 2 else d_g < high)
        values = residual[mask]
        if len(values) < 5:
            continue
        boots = np.asarray([values[rng.integers(0, len(values), len(values))].mean()
                            for _ in range(V1.B_BOOT)])
        q_low, q_high = np.percentile(boots, [2.5, 97.5])
        rows.append({"x": float((low + high) / 2), "y": float(values.mean()),
                     "lo": float(q_low), "hi": float(q_high), "n": len(values)})
    return rows


def one_dataset(name: str, records: list, cells: list) -> dict[str, np.ndarray]:
    fit = V1.fit_model(records, cells)
    d_g, residual, support, vertex, sd_y = components(records, fit)
    bins = fixed_bins(d_g, residual, np.random.default_rng(V1.SEED + 91))
    d_g0, residual0, _c1, _dgbar, _sd = fit_reduced(records, cells, "linear")
    bins0 = fixed_bins(d_g0, residual0, np.random.default_rng(V1.SEED + 91))
    if [b["x"] for b in bins] != [b["x"] for b in bins0]:
        raise ValueError(f"{name}: incompatible cache bins")
    x = np.linspace(*support, 300); central = tuple(np.quantile(d_g, [0.025, 0.975]))
    xc = np.linspace(*central, 300)
    edges = HIST_EDGES
    hist, _ = np.histogram(d_g, bins=edges, weights=np.full(len(d_g), 100.0 / len(d_g)))
    key = {"Huesken": "huesken", "Mixset": "mixset", "Taka": "taka"}[name]
    extra = {}
    if not support[0] <= vertex <= support[1]:
        endpoint = support[1] if vertex > support[1] else support[0]
        limit = min(vertex, -18.5) if vertex > support[1] else max(vertex, -50.5)
        xx = np.linspace(endpoint, limit, 100)
        extra = {f"{key}_x_extra": xx, f"{key}_y_extra": curve(fit, xx, sd_y),
                 f"{key}_vertex_y": np.asarray([curve(fit, np.asarray([vertex]), sd_y)[0]])}
    return {
        f"{key}_x_curve": x, f"{key}_y_curve": curve(fit, x, sd_y),
        f"{key}_x_central": xc, f"{key}_y_central": curve(fit, xc, sd_y),
        f"{key}_bin_x": np.asarray([b["x"] for b in bins]),
        f"{key}_bin_y": np.asarray([b["y"] for b in bins]),
        f"{key}_bin_lo": np.asarray([b["lo"] for b in bins]),
        f"{key}_bin_hi": np.asarray([b["hi"] for b in bins]),
        f"{key}_bin0_y": np.asarray([b["y"] for b in bins0])
        + float(np.mean(np.asarray([b["y"] for b in bins]) - np.asarray([b["y"] for b in bins0]))),
        f"{key}_hist_edges": edges, f"{key}_hist_pct": hist,
        f"{key}_vertex": np.asarray([vertex]), f"{key}_n": np.asarray([len(records)]),
        f"{key}_support": np.asarray(support), **extra,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=V1.results_directory() / "global_stability.npz")
    args = parser.parse_args(argv)
    if args.output.resolve() == (V1.RESULTS_DIR / "global_stability.npz").resolve() and (V1.B_BOOT, V1.SEED) != (4000, 0):
        raise ValueError("canonical cache requires B_BOOT=4000 and SEED=0")
    train, _ = V1.load_canonical_split()
    datasets = [("Huesken", train), ("Mixset", load_dataset("mix", data_dir=args.data_dir)),
                ("Taka", load_dataset("taka", data_dir=args.data_dir))]
    arrays: dict[str, np.ndarray] = {}
    for name, records in datasets:
        selected, *_ = V1.select_identity_terms(records)
        expected = list(V1.EXPECTED_CELLS) if name == "Huesken" else []
        if selected != expected:
            raise ValueError(f"{name}: selection {selected}, expected {expected}")
        arrays.update(one_dataset(name, records, selected))
    arrays["B_BOOT"] = np.asarray([V1.B_BOOT]); arrays["SEME"] = np.asarray([V1.SEED])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"wrote {args.output} ({len(arrays)} arrays; no row-level data retained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
