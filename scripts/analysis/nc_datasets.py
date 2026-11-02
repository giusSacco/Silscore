"""Load local Mix and Taka evaluation tables in source order with their published y labels."""
from __future__ import annotations

import csv
from pathlib import Path

from silscore.records import SiRNARecord, to_rna


ROOT = Path(__file__).resolve().parents[2]
NC_DIR = ROOT / "data" / "raw"
LICENCE = (
    "https://github.com/lulab/OligoFormer/blob/"
    "e2f53ad63387bbe166bf123949151e2bc9bf6ec3/LICENSE"
)
_FILES = {
    "mix": ("Mix.csv", None, "mixed published assays"),
    "taka": ("Taka.csv", None, "fluorescent reporter"),
}


def load_dataset(name: str, *, data_dir: Path | None = None):
    """Load local evaluation records in source order, retaining the published y label in extra."""
    name = name.lower()
    if name not in _FILES:
        raise ValueError(f"unknown dataset {name!r}; choose one of {sorted(_FILES)}")
    filename, cell_line, assay = _FILES[name]
    path = (data_dir or NC_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Acquire the evaluation file locally and pass --data-dir; "
            "automatic download is intentionally unavailable."
        )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no rows")
    missing = sorted({"siRNA", "mRNA", "label", "y"} - set(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    records = []
    for row_number, row in enumerate(rows, start=2):
        guide = to_rna(row["siRNA"].strip())
        if not guide or set(guide) - set("ACGU"):
            raise ValueError(f"{path}: row {row_number}: invalid guide")
        target = to_rna(row["mRNA"].strip().replace("X", "")) or None
        try:
            label = float(row["label"])
            y = int(row["y"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: row {row_number}: invalid label or y") from exc
        if y not in (0, 1):
            raise ValueError(f"{path}: row {row_number}: y must be 0 or 1")
        records.append(SiRNARecord(
            dataset=name,
            id=f"{name}_{row_number - 2}",
            guide=guide,
            efficacy=label,
            target=target,
            cell_line=cell_line,
            assay=assay,
            source_license=LICENCE,
            extra={"y": y},
        ))
    return records
