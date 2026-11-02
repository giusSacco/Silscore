

# %%
"""Prepare fixed-test cross-species comparison metrics."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import potency_modello_v1 as V1                              # noqa: E402
import dsir_reimplementation as D                           # noqa: E402
from bootstrap import correlation_ci, paired_correlation_ci                       # noqa: E402
from iscore import ARTICLE_ALPHA, FittedIScoreScorer         # noqa: E402
from rule_scorers import ReynoldsThermoScorer, UiTeiScorer   # noqa: E402

SEED, N_TRAIN = V1.SEED, 489
REPRESENTATIONS = ["sparse-19", "spectrum-19", "composite-19/19", "composite-21/21"]
MAIN_MODEL = V1.MODEL_NAME

def main() -> int:
    pool = V1.load_pool()
    recs = pool.recs
    seqs = [D.guide_with_overhang(r["guide"], r["overhang"]) for r in pool.rows]
    y = pool.y
    species = pool.species
    human_train, i_rodents, i_test = V1.fixed_target_split(species)
    print(f"fixed test: {len(i_test)} human; training: {N_TRAIN} rodent (new targets)")

    test = [recs[i] for i in i_test]
    y_te = y[i_test]

    def predictions(pool):

        train = [recs[i] for i in pool]
        seq_tr = [seqs[i] for i in pool]
        y_tr = y[pool]
        selected, *_ = V1.select_identity_terms(train)
        output = {}
        # Keep U1 fixed in both training groups.
        for name, terms in (("fisica pura", []), ("fisica + U1", [(1, "U")]),
                              (MAIN_MODEL, selected)):
            output[name] = np.asarray(V1.fit_model(train, terms).predict(test), float)
        for representation in REPRESENTATIONS:
            m = D.LarsLassoDSIR().fit(D.build_features(seq_tr, representation), y_tr)
            output[f"DSIR {representation}"] = np.asarray(
                m.predict(D.build_features([seqs[i] for i in i_test], representation)), float)
        iscore = FittedIScoreScorer(alpha=ARTICLE_ALPHA).fit(train, y_tr)
        output["i-Score riadattato"] = np.asarray(iscore.predict(test), float)
        for name, s in (("Reynolds 2004", ReynoldsThermoScorer()),
                        ("Ui-Tei 2004", UiTeiScorer())):
            output[name] = np.asarray(s.predict(test), float)
        return output, selected

    pred_same, selected_same = predictions(human_train)
    pred, selected = predictions(i_rodents)
    def equivalent(selected):
        if not selected:
            return "fisica pura"
        if selected == [(1, "U")]:
            return "fisica + U1"
        return None

    equivalent_new = equivalent(selected)
    equivalent_same = equivalent(selected_same)
    if equivalent_new or equivalent_same:
        print("  ⚠️ V1 matches a control model: "
              f"rodent={equivalent_new or 'none'}, "
              f"human={equivalent_same or 'none'}")

    rows = []
    for name, p in pred.items():
        r, lo, hi = correlation_ci(p, y_te)
        status = (f"identico a {equivalent_new}"
                 if name == MAIN_MODEL and equivalent_new else "")
        rows.append({"confronto": "pearson", "modello": name, "test": "specie (bersagli nuovi)",
                      "n": len(y_te), "valore": r, "ic_basso": lo, "ic_alto": hi,
                      "esito": status})
        rs, los, his = correlation_ci(pred_same[name], y_te)
        # Paired contrast on the same test records: rodent minus human.
        c, clo, chi = paired_correlation_ci(p, pred_same[name], y_te)
        drop_outcome = "esclude lo zero" if clo > 0 or chi < 0 else "include lo zero"
        if (name == MAIN_MODEL and equivalent_new
                and equivalent_new == equivalent_same):
            drop_outcome += f"; modello identico a {equivalent_new} nei due bracci"
        rows.append({"confronto": "caduta", "modello": name,
                      "test": "specie (altra meno stessa)", "n": len(y_te), "valore": c,
                      "ic_basso": clo, "ic_alto": chi,
                      "esito": drop_outcome})
        status_same = (f"identico a {equivalent_same}"
                        if name == MAIN_MODEL and equivalent_same else f"caduta {r - rs:+.3f}")
        rows.append({"confronto": "pearson", "modello": name, "test": "specie (stessa specie)",
                      "n": len(y_te), "valore": rs, "ic_basso": los, "ic_alto": his,
                      "esito": status_same})
        if name != MAIN_MODEL:
            d, dlo, dhi = paired_correlation_ci(pred[MAIN_MODEL], p, y_te)
            outcome = ("modelli identici" if name == equivalent_new else
                     "esclude lo zero" if dlo > 0 or dhi < 0 else "include lo zero")
            rows.append({"confronto": "contrasto", "modello": f"{MAIN_MODEL} - {name}",
                          "test": "specie (bersagli nuovi)", "n": len(y_te), "valore": d,
                          "ic_basso": dlo, "ic_alto": dhi, "esito": outcome})

    out = V1.result_path("potency_specie_per_figura.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["confronto", "modello", "test", "n", "valore",
                                           "ic_basso", "ic_alto", "esito"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
