"""RNA:RNA Watson–Crick energies at 37 °C, in kcal/mol.

Parameters: Xia et al. (1998), DOI 10.1021/bi9809425;
Turner & Mathews (2010), DOI 10.1093/nar/gkp892.
Sequences run 5′ to 3′ against their complementary strand. Duplex energy includes
initiation (4.09) and terminal AU penalties (0.45 each); stack profiles contain
only nearest-neighbor terms. Inputs must use uppercase RNA (ACGU).
"""

_WC_DG: dict[tuple[str, str], float] = {
    ("A", "A"): -0.93, ("A", "C"): -2.24, ("A", "G"): -2.08, ("A", "U"): -1.1,
    ("C", "A"): -2.11, ("C", "C"): -3.26, ("C", "G"): -2.36, ("C", "U"): -2.08,
    ("G", "A"): -2.35, ("G", "C"): -3.42, ("G", "G"): -3.26, ("G", "U"): -2.24,
    ("U", "A"): -1.33, ("U", "C"): -2.35, ("U", "G"): -2.11, ("U", "U"): -0.93,
}
_INIT_DG = 4.09
_AU_END_DG = 0.45
_RNA_BASES = frozenset("ACGU")


def _wc_dg37(seq5: str) -> float:
    if not seq5:
        return 0.0
    if not _RNA_BASES.issuperset(seq5):
        return float("nan")
    dg = _INIT_DG
    for end in (seq5[0], seq5[-1]):
        if end in ("A", "U"):
            dg += _AU_END_DG
    return dg + sum(_WC_DG[seq5[i], seq5[i + 1]] for i in range(len(seq5) - 1))


class TurnerNNBackend:
    """Compute duplex and per-stack energies for uppercase RNA guides."""

    def duplex_dg(self, guide: str) -> float:
        """Return duplex ΔG at 37 °C in kcal/mol, including initiation and ends."""
        return _wc_dg37(guide)

    def stack_profile(self, guide: str) -> list[float]:
        """Return one nearest-neighbor energy per adjacent pair, in guide order."""
        if len(guide) < 2:
            return []
        if not _RNA_BASES.issuperset(guide):
            return [float("nan")] * (len(guide) - 1)
        return [_WC_DG[guide[i], guide[i + 1]] for i in range(len(guide) - 1)]
