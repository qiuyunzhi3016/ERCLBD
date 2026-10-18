"""Create a reproducible 10% stratified test set from A3.jsonl."""

import json
import math
import random
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "datas" / "labels" / "A3.jsonl"
OUTPUT_FILE = BASE_DIR / "datas" / "test" / "365.jsonl"
SAMPLE_FRACTION = 0.1
RANDOM_SEED = 42
EXPECTED_LABELS = {"Decrease", "Increase", "Other"}


def main():
    rows_by_label = defaultdict(list)

    with INPUT_FILE.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {INPUT_FILE}"
                ) from error

            label = row.get("label")
            if label not in EXPECTED_LABELS:
                raise ValueError(
                    f"Unexpected label {label!r} on line {line_number}"
                )
            rows_by_label[label].append(row)

    if set(rows_by_label) != EXPECTED_LABELS:
        raise ValueError(
            f"Expected labels {sorted(EXPECTED_LABELS)}, "
            f"found {sorted(rows_by_label)}"
        )

    rng = random.Random(RANDOM_SEED)
    test_rows = []
    sampled_counts = {}
    for label in sorted(EXPECTED_LABELS):
        class_rows = rows_by_label[label]
        # Round half up so each class contributes as close to 10% as possible.
        sample_count = max(1, math.floor(len(class_rows) * SAMPLE_FRACTION + 0.5))
        test_rows.extend(rng.sample(class_rows, sample_count))
        sampled_counts[label] = sample_count

    rng.shuffle(test_rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as destination:
        for row in test_rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Input rows: {sum(map(len, rows_by_label.values()))}")
    print(f"Sampled per class: {sampled_counts}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
