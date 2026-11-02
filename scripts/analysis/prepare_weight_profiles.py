

# %%
"""Bootstrap the positional profiles used by the profile figures."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from silscore.model import _select_identity_terms
from nc_datasets import load_dataset          # noqa: E402
import potency_modello_v1 as V1                # noqa: E402

SEED = 20260821
DATASET = ("Huesken", "Mixset", "Taka", "Human-489", "Rodent-489")
MODELS = ("physics", "u1", "full")

def nucleotide_profile(w: np.ndarray) -> np.ndarray:

    w = np.asarray(w, float)
    return np.asarray([
        (w[p - 2] if p >= 2 else 0.0) + (w[p - 1] if p - 1 < len(w) else 0.0)
        for p in range(1, len(w) + 2)
    ])

def select_from_fit(physics, records) -> tuple[list, dict, int, float]:

    y = np.asarray([r.efficacy for r in records], float)
    residuals = y - np.asarray(physics.predict(records), float)
    return _select_identity_terms(residuals, records)

def fit_models(records):

    physics = V1.fit_model(records, [])
    selected, z, n_tested, threshold = select_from_fit(physics, records)
    u1 = V1.fit_model(records, [(1, "U")])
    full = u1 if selected == [(1, "U")] else V1.fit_model(records, selected)
    return physics, u1, full, selected, z, n_tested, threshold

def summarize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.median(x, axis=0), *np.percentile(x, [2.5, 97.5], axis=0)

def compute(name: str, records: list, n_boot: int, rng: np.random.Generator) -> dict:

    y = np.asarray([r.efficacy for r in records], float)
    sd_y = float(y.std(ddof=1))
    if not np.isfinite(sd_y) or sd_y == 0:
        raise ValueError(f"{name}: invalid efficacy standard deviation ({sd_y})")

    physics, u1, full, selected, z, n_tested, threshold = fit_models(records)
    fit = {"physics": physics, "u1": u1, "full": full}
    point = {}
    for model, fitted_model in fit.items():
        point[f"{model}_W"] = nucleotide_profile(fitted_model._parameters["wvec"])
        point[f"{model}_w"] = np.asarray(fitted_model._parameters["wvec"], float)

    boot_raw = {key: [] for key in point}
    boot_std = {key: [] for key in point}
    for _ in range(n_boot):
        idx = rng.integers(0, len(records), len(records))
        sample = [records[i] for i in idx]
        sd_b = float(np.asarray([r.efficacy for r in sample], float).std(ddof=1))
        if not np.isfinite(sd_b) or sd_b == 0:
            raise ValueError(f"{name}: bootstrap replicate has constant efficacy")
        f0 = V1.fit_model(sample, [])
        fu1 = V1.fit_model(sample, [(1, "U")])
        f1 = fu1 if selected == [(1, "U")] else V1.fit_model(sample, selected)
        replicate = {
            "physics_W": nucleotide_profile(f0._parameters["wvec"]),
            "physics_w": np.asarray(f0._parameters["wvec"], float),
            "u1_W": nucleotide_profile(fu1._parameters["wvec"]),
            "u1_w": np.asarray(fu1._parameters["wvec"], float),
            "full_W": nucleotide_profile(f1._parameters["wvec"]),
            "full_w": np.asarray(f1._parameters["wvec"], float),
        }
        for key, values in replicate.items():
            boot_raw[key].append(values)
            boot_std[key].append(values / sd_b)

    return {
        "name": name,
        "n": len(records),
        "sd_y": sd_y,
        "selected": selected,
        "z": z,
        "n_tested": n_tested,
        "threshold": threshold,
        "point_raw": point,
        "point_std": {k: v / sd_y for k, v in point.items()},
        "boot_raw": {k: np.asarray(v) for k, v in boot_raw.items()},
        "boot_std": {k: np.asarray(v) for k, v in boot_std.items()},
        "n_boot": n_boot,
    }

def write_csv(results: dict[str, dict], out: Path) -> None:
    fields = ["dataset", "n", "model", "scale", "coordinate", "index", "estimate",
              "bootstrap_median", "ci_low", "ci_high", "B"]
    with (out / "potency_weight_profiles_v1.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for name, r in results.items():
            for model in MODELS:
                for coord in ("W", "w"):
                    key = f"{model}_{coord}"
                    for scale, points, bootstrap in (
                        ("raw", r["point_raw"], r["boot_raw"]),
                        ("per_sd_efficacy", r["point_std"], r["boot_std"]),
                    ):
                        med, lo, hi = summarize(bootstrap[key])
                        for i, estimate in enumerate(points[key], 1):
                            writer.writerow({
                                "dataset": name, "n": r["n"], "model": model,
                                "scale": scale, "coordinate": coord, "index": i,
                                "estimate": f"{estimate:.10g}",
                                "bootstrap_median": f"{med[i-1]:.10g}",
                                "ci_low": f"{lo[i-1]:.10g}",
                                "ci_high": f"{hi[i-1]:.10g}", "B": r["n_boot"],
                            })

def main(argv: list[str]) -> int:
    n_boot = int(argv[1]) if len(argv) > 1 else V1.B_BOOT
    if n_boot < 20:
        raise ValueError("at least 20 bootstrap replicates are required for a diagnostic figure")
    train, _ = V1.load_canonical_split()
    pool = V1.load_pool()
    i_human_489, i_rodents, _ = V1.fixed_target_split(pool.species)
    datasets = {
        "Huesken": train,
        "Mixset": load_dataset("mix"),
        "Taka": load_dataset("taka"),
        "Human-489": [pool.recs[i] for i in i_human_489],
        "Rodent-489": [pool.recs[i] for i in i_rodents],
    }
    rng = np.random.default_rng(SEED)
    results = {}
    for name in DATASET:
        results[name] = compute(name, datasets[name], n_boot, rng)
        r = results[name]
        print(f"{name}: n={r['n']}, {r['n_tested']} cells, threshold={r['threshold']:.3f}")

    out = V1.result_path("potency_weight_profiles_v1.csv").parent
    write_csv(results, out)
    print(f"wrote {out / 'potency_weight_profiles_v1.csv'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
