"""Sequence records and explicit RNA conversion."""

from __future__ import annotations

from dataclasses import dataclass, field

_COMPLEMENT = str.maketrans("ACGUTacgut", "UGCAAUGCAA")


def to_rna(seq: str) -> str:
    """Uppercase a sequence and replace T with U, without validating its bases."""
    return seq.upper().replace("T", "U")


def reverse_complement(seq: str) -> str:
    """Return the 5′-to-3′ RNA reverse complement of an RNA or DNA sequence."""
    return seq.upper().translate(_COMPLEMENT)[::-1]


def check_rna(seq: str) -> str:
    """Require uppercase ACGU and return the sequence without converting it."""
    if not isinstance(seq, str) or not set(seq).issubset("ACGU"):
        raise ValueError("Expected uppercase RNA (ACGU); convert DNA explicitly with to_rna().")
    return seq


@dataclass(frozen=True)
class SiRNARecord:
    """A 5′-to-3′ guide, optional efficacy label, and source metadata."""

    dataset: str
    id: str
    guide: str
    efficacy: float | None = None
    passenger: str | None = None
    target: str | None = None
    cell_line: str | None = None
    assay: str | None = None
    source_license: str = ""
    use_verdict: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def sense(self) -> str:
        """Return the supplied passenger strand or the guide's reverse complement."""
        return self.passenger or reverse_complement(self.guide)
