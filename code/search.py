from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_search_config
from common.io import read_jsonl, write_jsonl
from common.mechanisms import candidate_file, candidate_mode, project_path

CONFIG_PATH = "configs/chca.json"


def run(config: dict) -> Path:
    """Search the configured Neo4j graph and write candidate mechanisms."""
    from py2neo import Graph

    mode = candidate_mode(config)
    output_path = candidate_file(config)
    groups = ("cf", "cr") if mode == "Causality" else ("cf", "cr", "broad")
    pairs = list(
        dict.fromkeys(
            tuple(pair)
            for group in groups
            for pair in config.get("relation_pairs", {}).get(group, [])
        )
    )
    graph = Graph(
        config["neo4j_uri"],
        auth=(
            config["neo4j_user"],
            os.getenv(
                "LBD_NEO4J_PASSWORD",
                config.get("neo4j_password", ""),
            ),
        ),
        name=config["neo4j_database"],
    )
    ensure_indexes(graph)

    all_results = []
    pubmed_index = build_pubmed_index(config["semrep_jsonl_path"])
    filter_entities = (
        []
        if mode == "All"
        else normalize_filter_entities(config.get("filter_entities", []))
    )
    filtered_count = 0

    for start_id, start_name in config["source_cuis"].items():
        for end_id, end_name in config["target_cuis"].items():
            results = query_two_hop_paths(
                graph,
                start_id,
                end_id,
                None if mode == "All" else config["allowed_entity_types"],
                None if mode == "All" else [list(pair) for pair in pairs],
                None,
            )
            results, removed_count = filter_paths(results, filter_entities)
            filtered_count += removed_count

            processed_results = []
            for row in results:
                pmid1 = normalize_pmid(row.get("pmid1"))
                pmid2 = normalize_pmid(row.get("pmid2"))
                evidence1 = extract_evidence(
                    pubmed_index.get(pmid1),
                    row.get("start_text", ""),
                    row.get("mid_text", ""),
                )
                evidence2 = extract_evidence(
                    pubmed_index.get(pmid2),
                    row.get("mid_text", ""),
                    row.get("end_text", ""),
                )
                row["pmid1_evidence"] = evidence1["evidence_text"]
                row["pmid2_evidence"] = evidence2["evidence_text"]
                row["pmid1"], row["pmid2"] = pmid1, pmid2
                processed_results.append(project_path(row))
            all_results.extend(processed_results)

    all_results = dedupe_paths(all_results)
    write_jsonl(output_path, all_results)
    if filter_entities:
        print(
            f"Filtered {filtered_count} paths with filter_entities; "
            f"saved {len(all_results)} {mode} paths to {output_path}"
        )
    else:
        print(f"Saved {len(all_results)} {mode} paths to {output_path}")
    return output_path


def normalize_entity_text(value: object) -> str:
    """Normalize an entity label for case-insensitive exact matching."""
    return " ".join(str(value or "").strip().lower().split())


def normalize_filter_entities(values) -> list[str]:
    """Normalize configured generic middle-entity labels."""
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return sorted(
        {
            normalized
            for value in values
            if (normalized := normalize_entity_text(value))
        }
    )


def entity_filter_keys(value: object) -> set[str]:
    """Build exact-match keys for an entity and its pipe-separated aliases."""
    normalized = normalize_entity_text(value)
    if not normalized:
        return set()
    keys = {normalized}
    keys.update(
        part.strip()
        for part in re.split(r"\s*\|\s*", normalized)
        if part.strip()
    )
    return keys


def filter_paths(
    rows: list[dict],
    filter_entities: list[str],
) -> tuple[list[dict], int]:
    """Remove paths whose middle entity is configured as generic."""
    if not filter_entities:
        return rows, 0

    blocked = set(filter_entities)
    kept = []
    removed = 0
    for row in rows:
        if entity_filter_keys(row.get("mid_text")) & blocked:
            removed += 1
        else:
            kept.append(row)
    return kept, removed


def ensure_indexes(graph) -> None:
    graph.run("CREATE INDEX mesh_name IF NOT EXISTS FOR (n:MESH) ON (n.name)")
    graph.run(
        "CREATE INDEX mesh_entity_type IF NOT EXISTS "
        "FOR (n:MESH) ON (n.entity_type)"
    )
    time.sleep(1)


def query_two_hop_paths(
    graph,
    start_id: str,
    end_id: str,
    allowed_entity_types: list[str] | None,
    relation_pairs: list[list[str]] | None,
    limit: int | None,
) -> list[dict]:
    query = """
    MATCH (a:MESH {name: $start}), (c:MESH {name: $end})
    MATCH (a)-[r1]->(mid:MESH)-[r2]->(c)
    WHERE ($allowed_entity_types IS NULL OR mid.entity_type IN $allowed_entity_types)
      AND ($relation_pairs IS NULL OR [type(r1), type(r2)] IN $relation_pairs)
    RETURN
        a.name          AS start_node,
        a.entity        AS start_text,
        mid.name        AS mid_node,
        mid.entity      AS mid_text,
        mid.entity_type AS mid_entity_type,
        c.name          AS end_node,
        c.entity        AS end_text,
        type(r1)        AS rel1,
        type(r2)        AS rel2,
        r1.pmid         AS pmid1,
        r2.pmid         AS pmid2
    """
    if limit is not None:
        query += "\nLIMIT $limit"
    cursor = graph.run(
        query,
        start=start_id,
        end=end_id,
        allowed_entity_types=allowed_entity_types,
        relation_pairs=relation_pairs,
        limit=limit,
    )
    return dedupe_paths(dict(record) for record in cursor)


def dedupe_paths(rows: list[dict]) -> list[dict]:
    seen = set()
    results = []
    for row in rows:
        key = (
            row.get("start_node"),
            row.get("mid_node"),
            row.get("end_node"),
            row.get("rel1"),
            row.get("rel2"),
            row.get("pmid1"),
            row.get("pmid2"),
        )
        if key not in seen:
            seen.add(key)
            results.append(row)
    return results


def build_pubmed_index(path: str) -> dict[str, dict]:
    """Index current top-level and older nested SemRep records by PMID."""
    index: dict[str, dict] = {}
    for row in read_jsonl(path, required=True):
        pmid = normalize_pmid(row.get("pmid"))
        document = row.get("document")
        if not isinstance(document, dict):
            document = row
        entities = row.get("entities")
        if not isinstance(entities, list):
            entities = document.get("entities", [])
        entity_index: dict[str, dict] = {}
        for entity in entities:
            entity_id = entity.get("id")
            if entity_id:
                entity_index[entity_id] = entity
                if entity_id.startswith("MESH:"):
                    entity_index[entity_id.split(":", 1)[1]] = entity
        index[pmid] = {
            "title": document.get("title", "") or "",
            "abstract": document.get("abstract", "") or "",
            "entities": entity_index,
        }
    return index


def extract_evidence(
    doc: dict | None,
    head: str,
    tail: str,
) -> dict:
    if not doc:
        return {
            "title": "",
            "abstract": "",
            "title_sentences": [],
            "abstract_sentences": [],
            "evidence_text": "",
        }
    title, abstract = doc.get("title", ""), doc.get("abstract", "")
    title_sentences = [
        sentence
        for sentence in split_sentences(title)
        if contains_entity(sentence, head) or contains_entity(sentence, tail)
    ]
    abstract_sentences = [
        sentence
        for sentence in split_sentences(abstract)
        if contains_entity(sentence, head) or contains_entity(sentence, tail)
    ]
    evidence_text = "\n".join(part for part in [title, abstract] if part).strip()
    return {
        "title": title,
        "abstract": abstract,
        "title_sentences": title_sentences,
        "abstract_sentences": abstract_sentences,
        "evidence_text": evidence_text,
    }


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?。！？])\s+", text)
        if item.strip()
    ]


def contains_entity(sentence: str, entity_text: str) -> bool:
    return bool(
        sentence
        and entity_text
        and entity_text.lower() in sentence.lower()
    )


def normalize_pmid(value) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search candidate mechanisms from the semantic knowledge graph."
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="Path to the case config JSON.",
    )
    args = parser.parse_args()
    run(load_search_config(args.config))


if __name__ == "__main__":
    main()
