"""Input checks for PDF extraction and saved benchmark predictions."""
import csv
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


extract = module("extract_huesken", "scripts/extract_huesken.py")
oligo = module("prepare_oligoformer", "scripts/analysis/prepare_oligoformer.py")


def test_unexpected_pdf_preserves_existing_csv(tmp_path, capsys):
    pdf, output = tmp_path / "table.pdf", tmp_path / "data.csv"
    pdf.write_bytes(b"not the reference supplement")
    output.write_bytes(b"existing data")
    with pytest.raises(SystemExit):
        extract.main(["--pdf", str(pdf), "--output", str(output)])
    assert "Unexpected PDF checksum" in capsys.readouterr().err
    assert output.read_bytes() == b"existing data"


def test_pdf_membership_marks_follow_columns(monkeypatch, tmp_path):
    words = [(10, 20, 20, 30, "12345"),
             (40, 20, 50, 30, "ACGUACGUACGUACGUACGuu"),
             (319, 20, 325, 30, "X"), (265, 20, 271, 30, "X"),
             (500, 20, 520, 30, "0.514")]

    class Document:
        def __enter__(self):
            return [types.SimpleNamespace(get_text=lambda kind: words)]

        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace(open=lambda p: Document()))
    row, = extract.parse(tmp_path / "table.pdf")
    assert row["record_id"] == "huesken_12345"
    assert row["guide"] == "ACGUACGUACGUACGUACG" and row["overhang"] == "uu"
    assert [key for key in extract.SPLITS if row[key]] == ["train_all", "train_rodent"]
    assert row["norm_activity"] == "0.514" and row["split"] == "train"


@pytest.fixture
def predictions(tmp_path):
    path = tmp_path / "predictions.csv"
    values = np.arange(78, dtype=float).reshape(3, 26)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "silscore_prediction", *oligo.PREDICTION_COLUMNS])
        writer.writerows((str(i), *row) for i, row in enumerate(values))
    return path, values


def test_prediction_columns_preserve_split_seed_order(predictions):
    path, expected = predictions
    np.testing.assert_array_equal(oligo.load_predictions(path, ["0", "1", "2"]), expected)
    assert oligo.PREDICTION_COLUMNS[5] == "split_44_seed_0"


def test_prediction_id_order_must_match_local_data(predictions):
    path, _ = predictions
    with pytest.raises(ValueError, match="IDs or row order"):
        oligo.load_predictions(path, ["1", "0", "2"])


@pytest.mark.parametrize("change, message", [
    (lambda text: text.replace("split_43_seed_0", "unknown"), "columns"),
    (lambda text: text.replace("0.0", "nan", 1), "prediction values"),
])
def test_invalid_prediction_tables_are_rejected(predictions, change, message):
    path, _ = predictions
    path.write_text(change(path.read_text()))
    with pytest.raises(ValueError, match=message):
        oligo.load_predictions(path, ["0", "1", "2"])
