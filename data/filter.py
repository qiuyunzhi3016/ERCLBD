"""Filter the source SemRep samples to the 12 predicates used in the paper."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "datas" / "original" / "all.jsonl"
DEFAULT_OUTPUT = ROOT / "datas" / "filter" / "all_12_predicates.jsonl"
EXPECTED_COUNT = 3_639

RELATIONS = frozenset(
    {
        "INHIBITS",
        "STIMULATES",
        "PRODUCES",
        "DISRUPTS",
        "AFFECTS",
        "INTERACTS_WITH",
        "AUGMENTS",
        "CAUSES",
        "PREDISPOSES",
        "TREATS",
        "PREVENTS",
        "ASSOCIATED_WITH",
    }
)


def filter_samples(input_path: Path, output_path: Path) -> tuple[int, Counter[str]]:
    """Write matching JSONL records without changing their serialized content."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_count = 0
    relation_counts: Counter[str] = Counter()
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source, temporary_path.open(
            "w", encoding="utf-8", newline=""
        ) as target:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {input_path}: {error}"
                    ) from error

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected a JSON object on line {line_number} of {input_path}."
                    )

                relation = str(record.get("relation", "")).strip().upper()
                if relation in RELATIONS:
                    target.write(raw_line if raw_line.endswith(("\n", "\r")) else raw_line + "\n")
                    selected_count += 1
                    relation_counts[relation] += 1

        if selected_count != EXPECTED_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_COUNT:,} samples for these predicates, "
                f"but found {selected_count:,}. The output was not replaced."
            )

        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return selected_count, relation_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter all.jsonl to the 12 selected relation predicates."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    args = parser.parse_args()

    count, relation_counts = filter_samples(args.input, args.output)
    print(f"Selected {count:,} samples from: {args.input}")
    for relation in sorted(RELATIONS):
        print(f"  {relation}: {relation_counts[relation]:,}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
