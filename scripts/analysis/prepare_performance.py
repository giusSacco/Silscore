

# %%
"""Compare Silscore with sequence and rule-based models."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import potency_modello_v1 as V1                              # noqa: E402
from bootstrap import correlation_ci, paired_correlation_ci       # noqa: E402
import dsir_reimplementation as D                           # noqa: E402
from iscore import ARTICLE_ALPHA, FittedIScoreScorer         # noqa: E402
from nc_datasets import load_dataset                         # noqa: E402
from rule_scorers import ReynoldsThermoScorer, UiTeiScorer   # noqa: E402

# Missing overhangs leave 21-nt scores unavailable for Mix and Taka.
DSIR_REPRESENTATIONS = ["sparse-19", "spectrum-19", "composite-19/19", "composite-21/21"]
DOMAIN_ONLY = {"composite-21/21"}

def main() -> int:
    train, test249 = V1.load_canonical_split()
    y_tr = np.array([r.efficacy for r in train])
    per_id = {r["record_id"]: D.guide_with_overhang(r["guide"], r["overhang"])
              for r in V1.load_rows()}
    seq21_tr = [per_id[r.id] for r in train]

    evaluations = {"249 (dominio)": test249, "mix": load_dataset("mix"), "taka": load_dataset("taka")}
    evaluations = {k: [r for r in v if r.efficacy is not None] for k, v in evaluations.items()}
    y = {k: np.asarray([r.efficacy for r in v], float) for k, v in evaluations.items()}
    seq_te = {"249 (dominio)": [per_id[r.id] for r in evaluations["249 (dominio)"]]}
    for k in ("mix", "taka"):
        seq_te[k] = [D.normalize(r.guide) for r in evaluations[k]]

    selected, *_ = V1.select_identity_terms(train)
    MAIN_MODEL = V1.MODEL_NAME
    first = V1.term_label(selected[0])
    pred = {}

    for name, terms in (("fisica pura", []), (f"fisica + {first}", selected[:1]),
                          (MAIN_MODEL, selected)):
        s = V1.fit_model(train, terms)
        pred[name] = {k: np.asarray(s.predict(v), float) for k, v in evaluations.items()}

    for representation in DSIR_REPRESENTATIONS:
        m = D.LarsLassoDSIR().fit(D.build_features(seq21_tr, representation), y_tr)
        pred[f"DSIR {representation}"] = {
            k: (None if (representation in DOMAIN_ONLY and k != "249 (dominio)")
                else m.predict(D.build_features(seq_te[k], representation)))
            for k in evaluations}

    iscore = FittedIScoreScorer(alpha=ARTICLE_ALPHA).fit(train, y_tr)
    pred["i-Score riadattato"] = {k: np.asarray(iscore.predict(v), float)
                                  for k, v in evaluations.items()}
    for name, s in (("Reynolds 2004", ReynoldsThermoScorer()), ("Ui-Tei 2004", UiTeiScorer())):
        pred[name] = {k: np.asarray(s.predict(v), float) for k, v in evaluations.items()}

    print("computing performance metrics")
    rows = []
    for name in pred:
        for k in evaluations:
            if pred[name][k] is None:
                continue
            r, lo, hi = correlation_ci(np.asarray(pred[name][k], float), y[k])
            rows.append({"confronto": "pearson", "modello": name, "test": k, "n": len(y[k]),
                          "valore": r, "ic_basso": lo, "ic_alto": hi, "esito": ""})

    for name in pred:
        if name == MAIN_MODEL:
            continue
        for k in evaluations:
            if pred[name][k] is None:
                continue
            d, lo, hi = paired_correlation_ci(np.asarray(pred[MAIN_MODEL][k], float),
                                    np.asarray(pred[name][k], float), y[k])
            outcome = "esclude lo zero" if lo > 0 or hi < 0 else "include lo zero"
            rows.append({"confronto": "contrasto", "modello": f"{MAIN_MODEL} - {name}", "test": k,
                          "n": len(y[k]), "valore": d, "ic_basso": lo, "ic_alto": hi,
                          "esito": outcome})

    out = V1.result_path("potency_vs_dsir.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["confronto", "modello", "test", "n", "valore",
                                           "ic_basso", "ic_alto", "esito"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
