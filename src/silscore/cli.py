"""Command-line interface for Silscore."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .model import Silscore, load_default_model
from .prediction import (
    normalize_target, predict_targets, read_prediction_input, read_records,
)


def build_parser() -> argparse.ArgumentParser:
    """Define the train and predict commands and their input options."""
    parser = argparse.ArgumentParser(
        description="Predict siRNA silencing efficacy or fit a Silscore model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Screen identities and fit a model.")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    predict = subparsers.add_parser(
        "predict", help="Predict silencing efficacy for target windows or supplied guides."
    )
    predict.add_argument("--model", type=Path, help="Model JSON; defaults to the bundled model.")
    inputs = predict.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--target", help="Direct RNA target sequence in 5-prime to 3-prime orientation.")
    inputs.add_argument("--input", type=Path, help="CSV with either target or guide columns.")
    predict.add_argument("--output", type=Path, default=Path("output/predictions.csv"))
    return parser


def main(argv=None) -> int:
    """Run training or prediction and write the model JSON or prediction CSV."""
    args = build_parser().parse_args(argv)
    if args.command == "train":
        records = read_records(args.input, training=True)
        model = Silscore().fit(records)
        model.save(args.output)
        print(f"Fitted {len(records)} records; {len(model.identity_terms)} identity terms.")
        return 0

    model = Silscore.load(args.model) if args.model else load_default_model()
    if args.target is not None:
        prediction_kind = "target"
        rows = predict_targets(model, [("target", normalize_target(args.target))])
    else:
        prediction_kind, input_rows = read_prediction_input(args.input)
        if prediction_kind == "target":
            rows = predict_targets(model, input_rows)
        else:
            records = input_rows
            scores = model.predict(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        if prediction_kind == "target":
            writer.writerow(["target_id", "start", "end", "target", "guide", "score"])
            writer.writerows(
                (row["target_id"], row["start"], row["end"], row["target"], row["guide"], row["score"])
                for row in rows
            )
            count = len(rows)
        else:
            writer.writerow(["id", "guide", "score"])
            writer.writerows(
                (record.id, record.guide, float(score)) for record, score in zip(records, scores)
            )
            count = len(records)
    print(f"Wrote {count} predictions to {args.output}")
    return 0
