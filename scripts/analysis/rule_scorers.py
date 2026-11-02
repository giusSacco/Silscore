

"""Implement the Reynolds and Ui-Tei rule comparators used in analyses."""
from __future__ import annotations

import numpy as np

REYNOLDS_MFE_THRESHOLD = -0.5  # kcal/mol
_COMPLEMENT = str.maketrans("ACGU", "UGCA")

class SeqfoldHairpin:

    def __init__(self, temp: float = 37.0) -> None:
        self.temp = temp

    def mfe(self, seq: str) -> float:
        import seqfold
        return float(seqfold.dg(seq, temp=self.temp))

def reynolds_score(sense: str, hairpin: SeqfoldHairpin | None = None,
                 mfe_threshold: float = REYNOLDS_MFE_THRESHOLD) -> int:

    s = sense
    score = 0
    gc = sum(b in "GC" for b in s) / len(s)
    if 0.30 <= gc <= 0.52:
        score += 1
    score += sum(b in "AU" for b in s[14:19])
    if hairpin is not None:
        if hairpin.mfe(s) >= mfe_threshold:
            score += 1
    else:
        kmers = [s[i:i + 9] for i in range(len(s) - 8)]
        if not any(nm.translate(_COMPLEMENT)[::-1] in s for nm in kmers):
            score += 1
    if s[18] == "A":
        score += 1
    if s[2] == "A":
        score += 1
    if s[9] == "U":
        score += 1
    if s[18] in "GC":
        score -= 1
    if s[12] == "G":
        score -= 1
    return score

class ReynoldsScorer:

    def __init__(self, hairpin: SeqfoldHairpin | None = None,
                 mfe_threshold: float = REYNOLDS_MFE_THRESHOLD) -> None:
        self._hairpin = hairpin
        self._mfe_threshold = mfe_threshold

    def fit(self, records, y=None) -> "ReynoldsScorer":
        return self

    def predict(self, records) -> np.ndarray:
        return np.array([reynolds_score(r.sense, self._hairpin, self._mfe_threshold)
                         for r in records], dtype=np.float64)

class ReynoldsThermoScorer(ReynoldsScorer):

    def __init__(self, mfe_threshold: float = REYNOLDS_MFE_THRESHOLD) -> None:
        super().__init__(hairpin=SeqfoldHairpin(), mfe_threshold=mfe_threshold)

def _has_long_gc_run(g: str, max_run: int = 9) -> bool:

    run_length = 0
    for b in g:
        if b in "GC":
            run_length += 1
            if run_length > max_run:
                return True
        else:
            run_length = 0
    return False

def uitei_score(guide: str) -> float:

    score = 0.0
    if guide[0] in "AU":
        score += 1
    if guide[18] in "GC":
        score += 1
    if sum(b in "AU" for b in guide[0:7]) >= 4:
        score += 1
    if not _has_long_gc_run(guide):
        score += 1
    return score

class UiTeiScorer:

    def fit(self, records, y=None) -> "UiTeiScorer":
        return self

    def predict(self, records) -> np.ndarray:
        return np.array([uitei_score(r.guide) for r in records], dtype=np.float64)
