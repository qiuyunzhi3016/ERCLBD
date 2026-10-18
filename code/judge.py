"""Classify two-hop candidate mechanisms and score evidence context."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from common.config import load_search_config
from common.mechanisms import candidate_file


CONFIG_PATH = "configs/fira.json"

DIRECTION_SIGNS = {"Increase": 1, "Decrease": -1}

OUTPUT_FIELDS = [
    "label",
    "score",
    "start_node",
    "start_text",
    "mid_node",
    "mid_text",
    "mid_entity_type",
    "end_node",
    "end_text",
    "rel1",
    "rel2",
    "pmid1",
    "pmid2",
    "pmid1_evidence",
    "pmid2_evidence",
]

DIRECTION_TEMPLATE = """
You are a biomedical relation classification assistant.
Classify the effect direction of the head entity on the tail entity based on the given evidence sentence.
Head entity: {head}
Tail entity: {tail}
Evidence sentence: {evidence}

\#\# Label definition: 
- Increase: the head entity promotes or enhances the tail entity. This includes increasing the expression, level, or activity of a substance; enhancing a biological function or process; or increasing the risk, severity, or progression of a disease, symptom, or adverse drug reaction. 
- Decrease: the head entity inhibits or reduces the tail entity. This includes decreasing the expression, level, or activity of a substance; suppressing a biological function or process; or reducing the risk, severity, or progression of a disease, symptom, or adverse drug reaction.
- Other: the evidence sentence only describes an association or co-occurrence between the entities, or provides insufficient evidence to determine the effect direction from the specified head entity to the tail entity.

\#\# Classification Steps: 
Step 1 (Initial Prediction): Identify the most likely label based on the evidence sentence.
Step 2 (Counterfactual Check): If the initial label is Increase, check whether Decrease is equally or more strongly supported. If the initial label is Decrease, check whether Increase is equally or more strongly supported. If the initial label is Other, check whether the sentence contains explicit or strongly implied directional evidence for Increase or Decrease.
Step 3 (Final Decision): Choose Increase or Decrease only when that direction is clearly and more strongly supported; otherwise, choose Other.

Return only one label: Increase, Decrease, or Other.


"""



CONTEXT_TEMPLATE = """
Evaluate the biological context consistency between two evidence sentences supporting the candidate mechanism.
Evidence 1 (PMID {pmid1}; {start} to {middle}): {evidence1}
Evidence 2 (PMID {pmid2}; {middle} to {end}): {evidence2}

\#\# Scoring criteria: 
Assess (1) species or experimental subjects, (2) anatomical, tissue, cellular, or subcellular locations, and (3) physiological or disease states and experimental conditions. Assign an overall score strictly between 0 and 1, higher scores indicate greater consistency.

Return only one numerical score between 0 and 1.

Candidate mechanism: {start} -[{rel1}]-> {middle} -[{rel2}]-> {end}
Mechanism label from the direction-combination rule: {label}


"""


def extract_abstract(evidence_text: object) -> str:
    """The candidate file stores title + newline + abstract in each evidence field."""
    text = str(evidence_text or "").strip()
    if "\n" in text:
        _, abstract = text.split("\n", 1)
        return abstract.strip()
    return text


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def contains_entity(sentence: str, entity_text: object) -> bool:
    entity = str(entity_text or "")
    return bool(sentence and entity and entity.lower() in sentence.lower())


def extract_relevant_sentences(
    evidence_text: object,
    head: object,
    tail: object,
) -> list[str]:
    """Keep abstract sentences mentioning either relation endpoint."""
    abstract = extract_abstract(evidence_text)
    return [
        sentence
        for sentence in split_sentences(abstract)
        if contains_entity(sentence, head) or contains_entity(sentence, tail)
    ]


def expected_path_sign(config: dict) -> int:
    """Return the expected net effect for the configured discovery task."""
    discovery_type = str(config.get("discovery_type", "Disease")).strip().lower()
    if discovery_type in {
        "adr",
        "adverse_reaction",
        "drug_adverse_reaction",
        "drug_adr",
    }:
        # For a drug-to-ADR task, the expected effect is to cause or worsen
        # the adverse reaction, so matching hop directions are Supporting.
        return 1
    if discovery_type in {"disease", "drug_disease", "disease_disease"}:
        # For drug-treatment-disease and disease-disease tasks, the expected
        # effect is to reduce or alleviate the target disease.
        return -1
    raise ValueError(f"Unsupported discovery_type {discovery_type!r}")


def parse_direction(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    text = text.strip().strip("`\"'").strip()
    match = re.fullmatch(r"(Increase|Decrease|Other)[.!]?", text, re.IGNORECASE)
    if not match:
        raise ValueError("Expected exactly one label: Increase, Decrease, or Other")
    return match.group(1).capitalize()


def parse_score(raw: str) -> float:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    text = text.strip().strip("`\"'").strip()
    if not re.fullmatch(r"(?:0?\.\d+|0|1(?:\.0+)?)", text):
        raise ValueError("Expected one numeric consistency score")
    score = float(text)
    if not 0 < score < 1:
        raise ValueError("Consistency score must be strictly between 0 and 1")
    return score


def resolve_api_key(config: dict) -> str:
    """Read the direct key, configured environment name, or legacy key value."""
    direct = str(config.get("llm_api_key") or "").strip()
    if direct:
        return direct
    configured = str(config.get("llm_api_key_env") or "").strip()
    if configured:
        from_environment = os.getenv(configured, "").strip()
        if from_environment:
            return from_environment
        if configured.startswith("sk-"):
            return configured
    return os.getenv("LBD_LLM_API_KEY", "").strip()


def build_client(config: dict):
    from openai import OpenAI

    key = resolve_api_key(config)
    if not key:
        raise ValueError(
            "No LLM API key found. Set llm_api_key, set the environment "
            "variable named by llm_api_key_env, or set LBD_LLM_API_KEY."
        )
    return OpenAI(
        api_key=key,
        base_url=config.get("llm_base_url"),
        timeout=float(config.get("llm_timeout", 60)),
    )


def request_parsed(client, prompt: str, config: dict, parser) -> object:
    retries = max(1, int(config.get("llm_retries", 3)))
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=config.get("llm_model", "deepseek-chat"),
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content or ""
            return parser(raw)
        except Exception as error:
            if attempt + 1 == retries:
                raise RuntimeError(
                    f"LLM request or response parsing failed after {retries} attempts"
                ) from error
            time.sleep(min(5, attempt + 1))
    raise RuntimeError("LLM request failed")


def build_direction_prompt(
    head: str,
    tail: str,
    relation: str,
    evidence_sentences: list[str],
) -> str:
    evidence = " ".join(evidence_sentences)
    if not evidence:
        evidence = "[No relevant sentence was found in the abstract; choose Other.]"
    return DIRECTION_TEMPLATE.format(
        head=head,
        tail=tail,
        relation=relation or "Unknown",
        evidence=evidence,
    )


def classify_relation(
    client,
    config: dict,
    head: str,
    tail: str,
    relation: str,
    evidence_sentences: list[str],
) -> str:
    prompt = build_direction_prompt(head, tail, relation, evidence_sentences)
    return request_parsed(client, prompt, config, parse_direction)


def build_context_prompt(path: dict, label: str, evidence1: str, evidence2: str) -> str:
    return CONTEXT_TEMPLATE.format(
        start=path.get("start_text", ""),
        middle=path.get("mid_text", ""),
        end=path.get("end_text", ""),
        rel1=path.get("rel1", ""),
        rel2=path.get("rel2", ""),
        label=label,
        pmid1=path.get("pmid1", ""),
        pmid2=path.get("pmid2", ""),
        evidence1=evidence1,
        evidence2=evidence2,
    )


def classify_mechanism(client, config: dict, path: dict, expected_sign: int):
    start = str(path.get("start_text", "") or "")
    middle = str(path.get("mid_text", "") or "")
    end = str(path.get("end_text", "") or "")

    sentences1 = extract_relevant_sentences(
        path.get("pmid1_evidence", ""), start, middle
    )
    sentences2 = extract_relevant_sentences(
        path.get("pmid2_evidence", ""), middle, end
    )

    direction1 = classify_relation(
        client,
        config,
        start,
        middle,
        str(path.get("rel1", "") or ""),
        sentences1,
    )
    direction2 = classify_relation(
        client,
        config,
        middle,
        end,
        str(path.get("rel2", "") or ""),
        sentences2,
    )

    if direction1 == "Other" or direction2 == "Other":
        return None

    path_sign = DIRECTION_SIGNS[direction1] * DIRECTION_SIGNS[direction2]
    label = "Supporting" if path_sign == expected_sign else "Opposing"
    evidence1 = " ".join(sentences1)
    evidence2 = " ".join(sentences2)
    score = request_parsed(
        client,
        build_context_prompt(path, label, evidence1, evidence2),
        config,
        parse_score,
    )

    result = {field: path.get(field, "") for field in OUTPUT_FIELDS}
    result.update(
        {
            "label": label,
            "score": score,
            "pmid1_evidence": evidence1,
            "pmid2_evidence": evidence2,
        }
    )
    return result


def write_output(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def run(config: dict) -> Path:
    source_path = candidate_file(config)
    output_path = candidate_file(config, classified=True)
    if not source_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {source_path}")

    with source_path.open("r", encoding="utf-8") as handle:
        paths = [json.loads(line) for line in handle if line.strip()]

    client = build_client(config) if paths else None
    retained = []
    rejected_other = 0
    path_sign = expected_path_sign(config)
    try:
        for index, path in enumerate(paths, 1):
            result = classify_mechanism(client, config, path, path_sign)
            if result is None:
                rejected_other += 1
            else:
                retained.append(result)

            if index % 25 == 0 or index == len(paths):
                print(
                    f"Judged {index}/{len(paths)}; "
                    f"retained {len(retained)}, removed {rejected_other}",
                    flush=True,
                )
    finally:
        if client:
            client.close()

    write_output(output_path, retained)
    print(f"Saved {len(retained)} mechanisms to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify candidate mechanisms for the configured case by "
            "two-hop effect direction and score evidence context."
        )
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="Path to the case config; also selects its candidate input/output files.",
    )
    args = parser.parse_args()
    run(load_search_config(args.config))


if __name__ == "__main__":
    main()



