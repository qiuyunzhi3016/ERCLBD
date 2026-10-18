
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
LABELS = ("Increase", "Decrease", "Other")


@dataclass
class Args:
    """All runtime configuration for this single-file classifier."""

    input_file: Path = field(default_factory=lambda: HERE / "datas" / "test" / "365.jsonl")
    output_dir: Path = field(default_factory=lambda: HERE / "output")
    output_file: Path | None = None
    llm_type: str = "llm"

    model_name: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "LLM_API_KEY"

    timeout: float = 60.0
    max_retries: int = 3
    retry_wait_seconds: float = 2.0

    resume: bool = True
    max_samples: int | None = None

    @classmethod
    def parse_args(cls, argv: list[str] | None = None) -> "Args":
        parser = argparse.ArgumentParser(
            description="Classify head-to-tail biomedical relation direction with an LLM."
        )
        parser.add_argument(
            "--input-file", type=Path, default=HERE / "datas" / "test" / "365.jsonl",
            help="Input JSONL file (default: datas/test/365.jsonl).",
        )
        parser.add_argument(
            "--output-dir", type=Path, default=HERE / "output",
            help="Directory for prediction JSONL files (default: output).",
        )
        parser.add_argument(
            "--output-file", type=Path, default=None,
            help="Optional explicit output path; otherwise output/<llm-type>.jsonl is used.",
        )
        parser.add_argument(
            "--llm-type", default="llm",
            help="Output filename stem, e.g. llm, gpt-4o, or deepseek (default: llm).",
        )
        parser.add_argument("--model-name", default="", help="Model identifier required by the API.")
        parser.add_argument(
            "--base-url", default="https://api.openai.com/v1",
            help="OpenAI-compatible API base URL (change it for other providers).",
        )
        parser.add_argument(
            "--api-key-env", default="LLM_API_KEY",
            help="Environment variable containing the API key (default: LLM_API_KEY).",
        )
        parser.add_argument("--timeout", type=float, default=60.0, help="API request timeout in seconds.")
        parser.add_argument("--max-retries", type=int, default=3, help="Total API attempts per sample.")
        parser.add_argument("--retry-wait-seconds", type=float, default=2.0)
        parser.add_argument("--max-samples", type=int, default=None, help="Optional input limit.")
        parser.add_argument("--no-resume", action="store_false", dest="resume", help="Start output from scratch.")
        parser.set_defaults(resume=True)

        parsed = parser.parse_args(argv)
        return cls(**vars(parsed))

    def resolved_input_file(self) -> Path:
        return resolve_path(self.input_file)

    def resolved_output_file(self) -> Path:
        if self.output_file is not None:
            return resolve_path(self.output_file)
        if not self.llm_type or Path(self.llm_type).name != self.llm_type:
            raise ValueError("llm_type must be a non-empty filename stem without directory components.")
        return resolve_path(self.output_dir) / f"{self.llm_type}.jsonl"

    def validate(self) -> None:
        if not self.model_name.strip():
            raise ValueError("Please provide --model-name.")
        if not self.base_url.strip():
            raise ValueError("Please provide --base-url for the OpenAI-compatible API.")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0.")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be greater than 0.")
        if self.retry_wait_seconds < 0:
            raise ValueError("retry_wait_seconds must be greater than or equal to 0.")
        if self.max_samples is not None and self.max_samples < 0:
            raise ValueError("max_samples must be greater than or equal to 0.")


def resolve_path(path: Path) -> Path:
    """Resolve relative paths against the directory containing this script."""
    return path if path.is_absolute() else HERE / path


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Read JSONL records, keeping their physical source line numbers."""
    if not path.is_file():
        raise FileNotFoundError(f"Input JSONL file does not exist: {path}")

    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL record must be an object.")
            for field_name in ("head", "tail", "text"):
                if field_name not in record or not isinstance(record[field_name], str):
                    raise ValueError(
                        f"{path}:{line_number}: expected a string field named {field_name!r}."
                    )
            records.append((line_number, record))
    return records


def read_processed_lines(path: Path) -> set[int]:
    """Read source line numbers already written by this script for resuming."""
    if not path.exists() or path.stat().st_size == 0:
        return set()

    processed: set[int] = set()
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each output record must be an object.")
            source_line = record.get("source_line")
            if isinstance(source_line, int):
                processed.add(source_line)
    return processed


def build_prompt(sample: dict[str, Any]) -> str:
    """Ask for the effect of head on tail and constrain the answer to one label."""
    return f"""
        You are a biomedical relation classification assistant.
        Classify the effect direction of the head entity on the tail entity based on the given evidence sentence.
        Head entity: {sample['head']}
        Tail entity: {sample['tail']}
        Evidence sentence: {sample['text']}.

        \#\# Label definition: 
        - Increase: the head entity promotes or enhances the tail entity. This includes increasing the expression, level, or activity of a substance; enhancing a biological function or process; or increasing the risk, severity, or progression of a disease, symptom, or adverse drug reaction. 
        - Decrease: the head entity inhibits or reduces the tail entity. This includes decreasing the expression, level, or activity of a substance; suppressing a biological function or process; or reducing the risk, severity, or progression of a disease, symptom, or adverse drug reaction.
        - Other: the evidence sentence only describes an association or co-occurrence between the entities, or provides insufficient evidence to determine the effect direction from the specified head entity to the tail entity.

        \#\# Classification Steps: 
        Step 1 (Initial Prediction): Identify the most likely label based on the evidence sentence.
        Step 2 (Counterfactual Check): If the initial label is Increase, check whether Decrease is equally or more strongly supported. If the initial label is Decrease, check whether Increase is equally or more strongly supported. If the initial label is Other, check whether the sentence contains explicit or strongly implied directional evidence for Increase or Decrease.
        Step 3 (Final Decision): Choose Increase or Decrease only when that direction is clearly and more strongly supported; otherwise, choose Other.

        Return only one of these labels: Increase, Decrease, Other.
        """


def normalize_label(raw_prediction: str | None) -> str:
    """Convert a model response to a label; use Other when it is ambiguous."""
    if not raw_prediction:
        return "Other"

    # Prefer a label at the start of the response, allowing simple Markdown or punctuation.
    leading_label = re.match(
        r"^\s*[`*_#\-\[({]*\s*(Increase|Decrease|Other)\b",
        str(raw_prediction),
        flags=re.IGNORECASE,
    )
    if leading_label:
        candidate = leading_label.group(1).lower()
        return {label.lower(): label for label in LABELS}[candidate]

    found = {
        match.group(1).lower()
        for match in re.finditer(r"(?<![A-Za-z])(Increase|Decrease|Other)(?![A-Za-z])", str(raw_prediction), re.I)
    }
    if len(found) == 1:
        candidate = found.pop()
        return {label.lower(): label for label in LABELS}[candidate]
    return "Other"


def get_api_key(args: Args) -> str:
    key_name = args.api_key_env.strip()
    api_key = os.environ.get(key_name, "") if key_name else ""
    if not api_key:
        raise ValueError(
            f"API key is missing. Set the {key_name or 'configured API key'} environment variable "
            "or pass its name with --api-key-env."
        )
    return api_key


def build_client(args: Args) -> Any:
    api_key = get_api_key(args)
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the 'openai' package to use OpenAI-compatible APIs.") from error

    return OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )


def call_api(client: Any, prompt: str, args: Args) -> str:
    """Send one prompt through an OpenAI-compatible Chat Completions API."""
    completion = client.chat.completions.create(
        model=args.model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content or ""


def call_with_retries(client: Any, prompt: str, args: Args) -> str:
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            return call_api(client, prompt, args)
        except Exception as error:
            last_error = error
            if attempt == args.max_retries:
                break
            wait_seconds = min(args.retry_wait_seconds * (2 ** (attempt - 1)), 30.0)
            print(
                f"API request failed ({attempt}/{args.max_retries}): {error}. "
                f"Retrying in {wait_seconds:g}s.",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"API request failed after {args.max_retries} attempts: {last_error}") from last_error


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args: Args) -> None:
    args.validate()
    input_path = args.resolved_input_file()
    output_path = args.resolved_output_file()
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")

    samples = read_jsonl(input_path)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        processed_lines = read_processed_lines(output_path)
    else:
        output_path.write_text("", encoding="utf-8")
        processed_lines = set()

    pending = [(line_no, sample) for line_no, sample in samples if line_no not in processed_lines]
    print(f"Input samples: {len(samples)}")
    print(f"Already processed: {len(samples) - len(pending)}")
    print(f"Pending samples: {len(pending)}")
    print(f"Model: {args.model_name}; API: {args.base_url}; output: {output_path}")

    if not pending:
        return

    client = build_client(args)
    for index, (line_no, sample) in enumerate(pending, start=1):
        if index == 1 or index % 10 == 0 or index == len(pending):
            print(f"Predicting {index}/{len(pending)}")

        raw_prediction = call_with_retries(client, build_prompt(sample), args)
        predicted_label = normalize_label(raw_prediction)
        output_record = dict(sample)
        output_record.update(
            {
                "gold_label": sample.get("label"),
                # Keeping the prediction in `label` lets the existing evaluate.py consume this file.
                "label": predicted_label,
                "predicted_label": predicted_label,
                "raw_prediction": raw_prediction,
                "llm_type": args.llm_type,
                "model_name": args.model_name,
                "source_line": line_no,
            }
        )
        append_jsonl(output_path, output_record)

    print(f"Predictions written to: {output_path}")


def main() -> int:
    args = Args.parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
