"""Validate prepared analysis CSVs and record their checksums."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


_HUESKEN_COLUMNS = (
    "record_id", "guide", "overhang", "norm_activity", "split",
    "train_all_human", "test_all_human", "train_rodent", "test_rodent",
)
_EVAL_FILES = ("Mix.csv", "Taka.csv")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    """Read a nonempty CSV with required columns and consistent row widths."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate columns")
        missing = [column for column in required if column not in fields]
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: malformed CSV row {reader.line_num}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def _number(value: str, *, path: Path, row_number: int, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: row {row_number}: {field} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{path}: row {row_number}: {field} is not finite")
    return parsed


def _flag(value: str, *, path: Path, row_number: int, field: str) -> int:
    if value not in {"0", "1"}:
        raise ValueError(f"{path}: row {row_number}: {field} must be 0/1, got {value!r}")
    return int(value)


def validate_huesken_csv(path: Path, *, strict_counts: bool = False) -> dict[str, int]:
    """Validate prepared Huesken rows and return total, train, and test counts.

    strict_counts also requires the canonical 2431 = 2182 + 249 partition.
    """
    rows = _read_csv(path, _HUESKEN_COLUMNS)
    identifiers = [row["record_id"] for row in rows]
    if any(not value.strip() for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path}: record_id values must be nonempty and unique")
    for row_number, row in enumerate(rows, start=2):
        guide = row["guide"]
        if len(guide) != 19 or set(guide) - set("ACGU"):
            raise ValueError(f"{path}: row {row_number}: expected an uppercase 19-nt RNA guide")
        _number(row["norm_activity"], path=path, row_number=row_number, field="norm_activity")
        if row["split"] not in {"train", "test"}:
            raise ValueError(f"{path}: row {row_number}: split must be train/test")
        for field in _HUESKEN_COLUMNS[5:]:
            _flag(row[field], path=path, row_number=row_number, field=field)
    counts = {
        "rows": len(rows),
        "train": sum(row["split"] == "train" for row in rows),
        "test": sum(row["split"] == "test" for row in rows),
    }
    if strict_counts and counts != {"rows": 2431, "train": 2182, "test": 249}:
        raise ValueError(f"canonical Huesken counts mismatch: {counts}")
    return counts


def validate_eval_csv(path: Path) -> dict[str, object]:
    """Check evaluation guides and labels without rewriting the source CSV."""
    rows = _read_csv(path, ("siRNA", "mRNA", "label", "y"))
    guides = []
    for row_number, row in enumerate(rows, start=2):
        guide = row["siRNA"].strip().upper().replace("T", "U")
        if set(guide) - set("ACGU") or not 15 <= len(guide) <= 30:
            raise ValueError(f"{path}: row {row_number}: invalid evaluation guide")
        _number(row["label"], path=path, row_number=row_number, field="label")
        _flag(row["y"], path=path, row_number=row_number, field="y")
        guides.append(guide)
    return {"rows": len(rows), "unique_guides": len(set(guides))}


def _inventory(root: Path, path: Path) -> dict[str, object]:
    return {"path": os.path.relpath(path, root), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def main(argv: list[str] | None = None) -> int:
    """Validate all three CSVs, optionally copy Huesken, and write an inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="repository root (default: parent of scripts/)")
    parser.add_argument("--raw-dir", type=Path,
                        help="directory containing Mix.csv and Taka.csv")
    parser.add_argument("--huesken-split", type=Path,
                        help="prepared canonical Huesken CSV (default: data/processed/huesken_canonical_split.csv)")
    parser.add_argument("--output", type=Path,
                        help="copy the validated CSV here (default: data/processed/huesken_canonical_split.csv)")
    parser.add_argument("--manifest", type=Path,
                        help="checksum manifest (default: data/cache/data_manifest.json)")
    parser.add_argument("--strict-canonical-counts", action="store_true",
                        help="require the published 2431 = 2182 + 249 row counts")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    canonical = root / "data/processed/huesken_canonical_split.csv"
    source = (args.huesken_split or canonical).resolve()
    output = (args.output or canonical).resolve()
    manifest = (args.manifest or root / "data/cache/data_manifest.json").resolve()
    raw = (args.raw_dir or root / "data/raw").resolve()
    eval_paths = [raw / name for name in _EVAL_FILES]
    input_paths = [source, *eval_paths]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        parser.error("missing prepared CSV inputs (see data/SOURCES.md):\n  " + "\n  ".join(missing))
    try:
        # Validate every input before copying data or writing the manifest.
        summaries = {"huesken_canonical_split": validate_huesken_csv(
            source, strict_counts=args.strict_canonical_counts)}
        summaries.update({path.name: validate_eval_csv(path) for path in eval_paths})
        inputs = [_inventory(root, path) for path in input_paths]
        if source != output:
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "local-only; no automatic download",
            "inputs": inputs,
            "outputs": [_inventory(root, output)],
            "summaries": summaries,
        }
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    print(f"Validated {len(input_paths)} input CSVs; wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
