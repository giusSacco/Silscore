"""Extract the canonical Huesken CSV from Supplementary Table 3."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PDF_SHA256 = "4d6cbaca93d98fa738659dfaf68875af3039b2061a94aa550b0c5a5b10e83d63"
CSV_SHA256 = "03508de5bb7a66208c7f3910e2d99300356c16fed88b0a93e72f2fff4ea6f074"
SPLITS = [
    "train_all", "train_all_human", "train_human_e2s", "train_rodent",
    "train_random_1091", "train_random_727", "train_random_545", "train_random_218",
    "test_all", "test_all_human", "test_human_e2s", "test_rodent",
]
COUNTS = [2182, 1744, 1229, 438, 1091, 727, 545, 218, 249, 198, 139, 51]
COLUMN_X = [265, 283, 301, 319, 337, 355, 373, 391, 410, 427, 445, 463]
GUIDE = re.compile(r"^([ACGU]{19})([acgtu]{2})$")


def parse(pdf: Path) -> list[dict]:
    """Read membership marks by their columns, retaining source row order."""
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError("PDF extraction requires PyMuPDF; see data/SOURCES.md.") from error
    rows = []
    with fitz.open(pdf) as document:
        for page in document:
            lines = collections.defaultdict(list)
            for x, y, _x1, _y1, token, *_ in page.get_text("words"):
                lines[round(y, 1)].append((x, token))
            for y in sorted(lines):
                words = sorted(lines[y])
                tokens = [token for _, token in words]
                if len(tokens) < 3 or not re.fullmatch(r"\d{3,}", tokens[0]):
                    continue
                guide = GUIDE.fullmatch(tokens[1])
                if not guide or not re.fullmatch(r"[01]\.\d{2,3}", tokens[-1]):
                    continue
                flags = [0] * len(SPLITS)
                for x, token in words[2:-1]:
                    column = min(range(len(COLUMN_X)), key=lambda i: abs(x - COLUMN_X[i]))
                    if token != "X" or abs(x - COLUMN_X[column]) > 6:
                        raise ValueError(f"NAS {tokens[0]}: unexpected membership mark")
                    flags[column] = 1
                rows.append(dict(
                    nas_id=tokens[0], record_id=f"huesken_{tokens[0]}",
                    guide=guide[1], overhang=guide[2], norm_activity=tokens[-1],
                    split="train" if flags[0] else "test", **dict(zip(SPLITS, flags)),
                ))
    return rows


def canonical_csv(rows: list[dict]) -> bytes:
    """Check the published memberships and serialize the exact reference CSV."""
    if len(rows) != 2431 or len({r["nas_id"] for r in rows}) != 2431:
        raise ValueError("Expected 2431 unique Huesken records")
    if [sum(r[key] for r in rows) for key in SPLITS] != COUNTS:
        raise ValueError("Extracted split counts differ from the reference")
    if any(r["train_all"] + r["test_all"] != 1 for r in rows):
        raise ValueError("Training and test must partition the records")
    stream = io.StringIO(newline="")
    fields = ["nas_id", "record_id", "guide", "overhang", "norm_activity", "split", *SPLITS]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    data = stream.getvalue().encode("utf-8")
    if hashlib.sha256(data).hexdigest() != CSV_SHA256:
        raise ValueError("Extracted CSV differs from the canonical reference")
    return data


def main(argv=None) -> int:
    """Require the reference PDF and write the verified canonical CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True, help="local Supplementary Table 3 PDF")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/processed/huesken_canonical_split.csv")
    args = parser.parse_args(argv)
    try:
        if hashlib.sha256(args.pdf.read_bytes()).hexdigest() != PDF_SHA256:
            raise ValueError("Unexpected PDF checksum; obtain Supplementary Table 3 from data/SOURCES.md")
        data = canonical_csv(parse(args.pdf))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output} (2431 records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
