#!/usr/bin/env python3
"""Calculate per-class F1 and macro-F1 from two JSONL files' ``label`` fields."""

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = HERE / "output" / "llm.jsonl"
DEFAULT_GOLD = HERE / "datas" / "test" / "365.jsonl"


def read_labels(path: Path) -> list[str]:
    labels = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict) or "label" not in record:
                raise ValueError(f"{path}:{line_number}: expected a JSON object with a 'label' field")
            label = record["label"]
            if not isinstance(label, str):
                raise ValueError(f"{path}:{line_number}: 'label' must be a string")
            labels.append(label)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate JSONL predictions using the 'label' field in each file."
    )
    parser.add_argument("--pred", type=Path, default=DEFAULT_PREDICTIONS,
                        help=f"prediction JSONL (default: {DEFAULT_PREDICTIONS})")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD,
                        help=f"ground-truth JSONL (default: {DEFAULT_GOLD})")
    args = parser.parse_args()

    try:
        predicted = read_labels(args.pred)
        actual = read_labels(args.gold)
        if len(predicted) != len(actual):
            raise ValueError(
                f"Different record counts: predictions={len(predicted)}, gold={len(actual)}"
            )
        if not actual:
            raise ValueError("The input files contain no labeled records")

        classes = sorted(set(actual) | set(predicted))
        print(f"Records: {len(actual)}")
        print("Per-class metrics (classes are the union of gold and predicted labels):")
        print(f"{'Label':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")

        precision_scores = []
        recall_scores = []
        f1_scores = []
        for label in classes:
            true_positive = sum(gold == label and pred == label
                                for gold, pred in zip(actual, predicted))
            false_positive = sum(gold != label and pred == label
                                 for gold, pred in zip(actual, predicted))
            false_negative = sum(gold == label and pred != label
                                 for gold, pred in zip(actual, predicted))
            support = sum(gold == label for gold in actual)

            precision_denominator = true_positive + false_positive
            recall_denominator = true_positive + false_negative
            precision = true_positive / precision_denominator if precision_denominator else 0.0
            recall = true_positive / recall_denominator if recall_denominator else 0.0
            f1_denominator = 2 * true_positive + false_positive + false_negative
            f1 = 2 * true_positive / f1_denominator if f1_denominator else 0.0
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            print(f"{label:<30} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")

        macro_precision = sum(precision_scores) / len(precision_scores)
        macro_recall = sum(recall_scores) / len(recall_scores)
        macro_f1 = sum(f1_scores) / len(f1_scores)
        print(f"Macro-Precision: {macro_precision:.4f}")
        print(f"Macro-Recall: {macro_recall:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
