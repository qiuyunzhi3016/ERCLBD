"""Rank intermediate concepts with the Chapter 5 reliability equations."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from common.config import load_search_config
from common.io import read_jsonl, write_jsonl
from common.mechanisms import candidate_file


CONFIG_PATH = "configs/chca.json"
EPSILON = 1e-8
LITERATURE_ALPHA = 1.0
SORT_VARIANTS = {"full", "oe", "cs", "lw"}

def resolve_sort_variant(config: dict) -> tuple[str, Path]:
    variant = normalize_sort_variant(config.get("variant", "full"))

    output = Path(config["sort_output_path"])
    if variant != "full":
        output = output.with_name(
            f"{output.stem}-{variant}{output.suffix}"
        )
    return variant, output


def run(config: dict) -> Path:
    variant, output_path = resolve_sort_variant(config)
    source = candidate_file(config, classified=True)
    classified_paths = read_jsonl(source, required=True)
    if any(
        path.get("label") not in {"Supporting", "Opposing", "Other"}
        for path in classified_paths
    ):
        raise ValueError("Invalid label in classified input")

    # Other mechanisms do not participate in any subsequent calculation.
    paths = []
    for path in classified_paths:
        if path["label"] == "Other":
            continue
        reliability = context_reliability(path, source)
        paths.append(
            dict(
                path,
                source_file=str(source),
                R=reliability,
                context_reliability=reliability,
            )
        )

    # w/o OE removes opposing mechanisms and all of their evidence statistics
    # before the per-middle-concept aggregation is performed.
    if variant == "oe":
        paths = [path for path in paths if path["label"] == "Supporting"]

    ranked = aggregate_by_middle(paths, config, variant)
    write_jsonl(output_path, ranked)
    print(
        f"Saved {len(ranked)} ranked middle concepts to "
        f"{output_path}"
    )
    return output_path



def normalize_sort_variant(value: object) -> str:
    variant = str(value or "full").strip().lower()
    if variant not in SORT_VARIANTS:
        raise ValueError(
            f"Unsupported sort variant {variant!r}; "
            "choose one of: full, oe, cs, lw"
        )
    return variant


def context_reliability(path: dict, source: Path | str = "") -> float:
    """Return R(pi) in [0, 1] from a normalized score or context dimensions.

    Current ``bs-filter.jsonl`` files store the reliability directly in
    ``score`` using the open interval (0, 1).
    """
    score = path.get("score")
    if (
        not isinstance(score, bool)
        and isinstance(score, (int, float))
        and math.isfinite(score)
        and 0 < score < 1
    ):
        # Prefer the evidence-based score when auxiliary fields are present.
        return float(score)

    context = path.get("context")
    if context not in (None, ""):
        if not isinstance(context, (list, tuple)) or len(context) != 3:
            raise ValueError(
                f"Invalid context dimensions in {source}: {context!r}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value not in {0, 0.5, 1}
            for value in context
        ):
            raise ValueError(
                f"Context dimensions must each be 0, 0.5, or 1 in {source}"
            )
        return math.fsum(context) / 3.0

    for key in ("context_reliability", "R"):
        if key in path and path[key] not in (None, ""):
            value = path[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    f"Invalid {key} in {source}: {value!r}; expected [0, 1]"
                )
            return float(value)

    raise ValueError(
        f"Invalid score in {source}: {score!r}; expected a finite number "
        "in (0, 1)"
    )


def normalize_discovery_type(value: object) -> str:
    normalized = str(value or "Disease").strip().lower()
    if normalized in {
        "adr",
        "adverse_reaction",
        "drug_adverse_reaction",
        "drug_adr",
    }:
        return "ADR"
    if normalized in {"disease", "drug_disease", "disease_disease"}:
        return "Disease"
    raise ValueError(
        f"Unsupported discovery_type {value!r}; choose ADR or Disease"
    )


def normalize_pmid(value) -> str:
    if isinstance(value, list):
        value = value[0] if value else None
    return "" if value is None else str(value).strip()


def pmid_set(value) -> set[str]:
    """Keep every PMID when an evidence field contains a list."""
    values = value if isinstance(value, list) else [value]
    return {
        pmid for item in values
        if (pmid := normalize_pmid(item))
    }


def evidence_statistics(paths: list[dict], variant: str) -> dict:
    """Aggregate mean consistency and deduplicated literature coverage.

    m = min(N_ab, N_bc), U = |PMID_ab union PMID_bc|
    N_eff = m * (1 + alpha * (U - m) / (U + m))
    L = ln(1 + N_eff), S = L * mean(score).
    """
    ab_pmids: set[str] = set()
    bc_pmids: set[str] = set()
    for path in paths:
        ab_pmids.update(pmid_set(path.get("pmid1")))
        bc_pmids.update(pmid_set(path.get("pmid2")))

    all_pmids = sorted(ab_pmids | bc_pmids)
    n_ab = len(ab_pmids)
    n_bc = len(bc_pmids)
    n = min(n_ab, n_bc)
    union_count = len(all_pmids)
    effective_count = (
        n * (
            1.0 + LITERATURE_ALPHA
            * (union_count - n) / (union_count + n)
        )
        if n else 0.0
    )
    raw_literature_strength = math.log1p(effective_count)
    reliability_values = [
        context_reliability(path, path.get("source_file", ""))
        for path in paths
    ]
    score_sum = math.fsum(reliability_values)
    average_reliability = (
        score_sum / len(paths) if paths else None
    )

    # Empty directions contribute zero in every variant.
    # CS removes consistency weighting; LW removes literature weighting.
    consistency = (
        (1.0 if variant == "cs" else average_reliability)
        if paths else 0.0
    )
    literature_strength = (
        (1.0 if variant == "lw" else raw_literature_strength)
        if paths else 0.0
    )
    return {
        "ab_pmids": sorted(ab_pmids),
        "bc_pmids": sorted(bc_pmids),
        "all_pmids": all_pmids,
        "N_ab": n_ab,
        "N_bc": n_bc,
        "N": n,
        "U": union_count,
        "N_eff": effective_count,
        "L_raw": raw_literature_strength,
        "L": literature_strength,
        "R_bar": average_reliability,
        "C": consistency,
        "score_sum": score_sum,
        "L_tilde": literature_strength * consistency,
    }


def aggregate_by_middle(
    paths: list[dict],
    config: dict,
    variant: str | None = None,
) -> list[dict]:
    variant = normalize_sort_variant(
        config.get("variant", "full") if variant is None else variant
    )
    if variant == "oe":
        paths = [
            path for path in paths if path.get("label") == "Supporting"
        ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        if path.get("label") not in {"Supporting", "Opposing"}:
            continue
        grouped[path.get("mid_node") or path.get("mid_text", "")].append(path)

    rows = []
    for mid_node, group in grouped.items():
        positive = [
            path for path in group if path.get("label") == "Supporting"
        ]
        negative = [
            path for path in group if path.get("label") == "Opposing"
        ]
        # Rank only concepts with at least one supporting mechanism.
        if not positive:
            continue

        positive_stats = evidence_statistics(positive, variant)
        negative_stats = evidence_statistics(negative, variant)
        adjusted_positive = positive_stats["L_tilde"]
        adjusted_negative = negative_stats["L_tilde"]
        evidence_balance = adjusted_positive / (
            adjusted_positive + adjusted_negative + EPSILON
        )
        final_score = math.log1p(adjusted_positive) * evidence_balance
        representative = group[0]
        row = {
            "rank": None,
            "variant": variant,
            "discovery_type": normalize_discovery_type(
                config.get("discovery_type", "Disease")
            ),
            "mid_node": mid_node,
            "mid_text": representative.get("mid_text", ""),
            "mid_entity_type": representative.get(
                "mid_entity_type", ""
            ),
            "start_node": representative.get("start_node", ""),
            "start_text": representative.get("start_text", ""),
            "end_node": representative.get("end_node", ""),
            "end_text": representative.get("end_text", ""),
            "final_score": final_score,
            "evidence_balance": evidence_balance,
            # Each direction combines literature coverage with mean
            # mechanism consistency: S+/- = L+/- * C+/-.
            "S_positive": adjusted_positive,
            "S_negative": adjusted_negative,
            "epsilon": EPSILON,
            "literature_alpha": LITERATURE_ALPHA,
            "positive_path_count": len(positive),
            "negative_path_count": len(negative),
            "positive_unique_pmids": positive_stats["all_pmids"],
            "negative_unique_pmids": negative_stats["all_pmids"],
            "positive_ab_unique_pmids": positive_stats["ab_pmids"],
            "positive_bc_unique_pmids": positive_stats["bc_pmids"],
            "negative_ab_unique_pmids": negative_stats["ab_pmids"],
            "negative_bc_unique_pmids": negative_stats["bc_pmids"],
            "positive_paths": [summarize_path(path) for path in positive],
            "negative_paths": [summarize_path(path) for path in negative],
        }
        row.update(evidence_output_fields("positive", positive_stats))
        row.update(evidence_output_fields("negative", negative_stats))
        rows.append(row)

    # The paper specifies ranking by S_final only. Python's stable sort keeps
    # the input order for exact ties without adding another scoring criterion.
    rows.sort(key=lambda row: row["final_score"], reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def evidence_output_fields(prefix: str, stats: dict) -> dict:
    """Expose the evidence counts, reliability mean and adjusted strength."""
    return {
        f"N_ab_{prefix}": stats["N_ab"],
        f"N_bc_{prefix}": stats["N_bc"],
        f"N_{prefix}": stats["N"],
        f"U_{prefix}": stats["U"],
        f"N_eff_{prefix}": stats["N_eff"],
        f"L_raw_{prefix}": stats["L_raw"],
        f"L_{prefix}": stats["L"],
        f"C_{prefix}": stats["C"],
        f"score_sum_{prefix}": stats["score_sum"],
        f"R_bar_{prefix}": stats["R_bar"],
        f"L_tilde_{prefix}": stats["L_tilde"],
    }


def summarize_path(path: dict) -> dict:
    keys = [
        "label",
        "start_node",
        "start_text",
        "mid_node",
        "mid_text",
        "end_node",
        "end_text",
        "rel1",
        "rel2",
        "pmid1",
        "pmid2",
        "score",
        "R",
        "context_reliability",
        "evidence_reason",
        "source_file",
        "pmid1_evidence",
        "pmid2_evidence",
    ]
    return {key: path.get(key) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="Path to the case config JSON.",
    )
    args = parser.parse_args()
    config = load_search_config(args.config)
    # load_search_config currently normalizes the search variant to "full".
    # Restore the sort-only variant from the user config without changing the
    # shared search/judge configuration flow.
    raw_config_path = Path(args.config)
    if not raw_config_path.is_absolute():
        raw_config_path = Path(__file__).resolve().parent / raw_config_path
    with raw_config_path.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    config["variant"] = raw_config.get("variant", "full")
    run(config)


if __name__ == "__main__":
    main()
