"""Input parsing and target-window prediction."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

from .records import SiRNARecord, reverse_complement

_RNA_BASES = frozenset("ACGU")


def read_records(path, training: bool = False) -> list[SiRNARecord]:
    """Read guide records from CSV, using efficacy or norm_activity for training.

    If training and a split column is present, retain only train rows.
    Guide sequences must already use uppercase RNA.
    """
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("Duplicate CSV columns are not allowed.")
        if "guide" not in fields:
            raise ValueError("Input CSV requires a guide column.")
        if "target" in fields:
            raise ValueError("Input CSV cannot contain both target and guide columns.")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {reader.line_num}: expected {len(fields)} fields.")
            rows.append(row)
    if not rows:
        raise ValueError("Input CSV must contain at least one row.")
    if training and rows and "split" in rows[0]:
        if not {row["split"] for row in rows}.issubset({"train", "test"}):
            raise ValueError("Unknown split value; expected train/test.")
        rows = [row for row in rows if row["split"] == "train"]
    records = []
    for index, row in enumerate(rows):
        value = row.get("efficacy", row.get("norm_activity"))
        if training and value is None:
            raise ValueError("Training CSV requires efficacy or norm_activity.")
        records.append(SiRNARecord(
            dataset="input", id=row.get("record_id", row.get("id", str(index))),
            guide=row["guide"], efficacy=float(value) if training else None,
        ))
    return records


def normalize_target(sequence: str) -> str:
    """Validate a target of at least 19 nt and return uppercase RNA.

    Remove whitespace and convert T to U, warning on stderr about each change.
    """
    if not isinstance(sequence, str):
        raise ValueError("Target must be text.")
    unspaced = "".join(sequence.split())
    compact = unspaced.upper()
    normalized = compact.replace("T", "U")
    if len(normalized) < 19:
        raise ValueError(f"Target must be at least 19 nt after normalization; got {len(normalized)}.")
    invalid = sorted(set(normalized) - _RNA_BASES)
    if invalid:
        raise ValueError(f"Target contains invalid or ambiguous bases: {', '.join(invalid)}.")
    changes = []
    if unspaced != sequence:
        changes.append("removed whitespace")
    if compact != unspaced:
        changes.append("converted lowercase to uppercase")
    if "T" in compact:
        changes.append("converted T to U (treated as RNA)")
    if changes:
        print(f"Warning: target normalized: {', '.join(changes)}.", file=sys.stderr)
    return normalized


def read_target_csv(path) -> list[tuple[str, str]]:
    """Read (id, normalized target) pairs; missing IDs use the zero-based row index."""
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("Duplicate CSV columns are not allowed.")
        if "target" not in fields:
            raise ValueError("Target CSV requires a target column.")
        if "guide" in fields:
            raise ValueError("Target CSV cannot contain a guide column.")
        rows = list(reader)
    if not rows:
        raise ValueError("Target CSV must contain at least one row.")
    targets = []
    for index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed target CSV row {index + 2}.")
        targets.append((row.get("id") or str(index), normalize_target(row.get("target", ""))))
    return targets


def read_prediction_input(path) -> tuple[str, object]:
    """Return the input kind (target or guide) and its parsed CSV records."""
    with Path(path).open(newline="") as stream:
        fields = csv.DictReader(stream).fieldnames or []
    if len(fields) != len(set(fields)):
        raise ValueError("Duplicate CSV columns are not allowed.")
    if "target" in fields:
        if "guide" in fields:
            raise ValueError("Input CSV cannot contain both target and guide columns.")
        return "target", read_target_csv(path)
    if "guide" in fields:
        return "guide", read_records(path)
    raise ValueError("Input CSV requires either a target or a guide column.")


def target_records(targets: list[tuple[str, str]]) -> tuple[list[SiRNARecord], list[dict]]:
    """Build reverse-complement guides for all 19-nt target windows.

    Return guide records and window metadata in input order, with 1-based
    inclusive target coordinates and a step of one nucleotide.
    """
    records, rows = [], []
    for target_id, target in targets:
        try:
            target = normalize_target(target)
        except ValueError as error:
            raise ValueError(f"Target {target_id!r}: {error}") from error
        for start0 in range(len(target) - 18):
            window = target[start0:start0 + 19]
            guide = reverse_complement(window)
            records.append(SiRNARecord(dataset="target", id=f"{target_id}:{start0 + 1}", guide=guide))
            rows.append({
                "target_id": target_id, "start": start0 + 1, "end": start0 + 19,
                "target": window, "guide": guide,
            })
    return records, rows


def predict_targets(model, targets: list[tuple[str, str]]) -> list[dict]:
    """Return target-window metadata with one model score per 19-nt guide."""
    records, rows = target_records(targets)
    scores = model.predict(records)
    if len(scores) != len(rows):
        raise RuntimeError(f"Model returned {len(scores)} scores for {len(rows)} target windows.")
    return [dict(row, score=float(score)) for row, score in zip(rows, scores)]
