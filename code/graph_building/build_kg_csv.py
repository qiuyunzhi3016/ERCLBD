from __future__ import annotations

import csv
from pathlib import Path

from common.io import read_jsonl
from common.progress import tqdm


UNDIRECTED_RELATIONS = {
    "ASSOCIATED_WITH",
    "INTERACTS_WITH",
    "COEXISTS_WITH",
}


def run(config: dict) -> tuple[Path, Path]:
    kg_csv = Path(config["kg_csv_path"])
    nodes_csv = Path(config["nodes_csv_path"])
    rels_csv = Path(config["relationships_csv_path"])

    if nodes_csv.exists() and rels_csv.exists() and nodes_csv.stat().st_size > 0 and rels_csv.stat().st_size > 0:
        print(f"Neo4j CSV files already exist: {nodes_csv}, {rels_csv}")
        return nodes_csv, rels_csv

    relationship_count = process_jsonl_to_csv(config["semrep_jsonl_path"], kg_csv)
    prepare_neo4j_csvs_streaming(kg_csv, nodes_csv, rels_csv, relationship_count)
    return nodes_csv, rels_csv


def process_jsonl_to_csv(input_file: str, output_file: str | Path) -> int:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(input_file, required=True)

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "pmid",
            "relation_type",
            "arg1_id",
            "arg1_text",
            "arg1_type",
            "arg2_id",
            "arg2_text",
            "arg2_type",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        relationship_count = 0
        for entry in tqdm(
            rows,
            desc="Convert SemRep to KG CSV",
            unit="record",
            unit_scale=True,
            dynamic_ncols=True,
        ):
            entity_map = {}
            for entity in entry.get("entities", []):
                entity_id = normalize_entity_id(entity.get("id"))
                entity_map[entity_id] = {"text": entity.get("text", ""), "type": entity.get("type", "")}

            for relation in entry.get("relations", []):
                arg1 = normalize_entity_id(relation.get("arg1"))
                arg2 = normalize_entity_id(relation.get("arg2"))
                arg1_info = entity_map.get(arg1, {"text": "", "type": ""})
                arg2_info = entity_map.get(arg2, {"text": "", "type": ""})
                rel_type = str(relation.get("type", "")).strip().upper()
                row = {
                    "pmid": entry.get("pmid", ""),
                    "relation_type": rel_type,
                    "arg1_id": arg1,
                    "arg1_text": arg1_info["text"],
                    "arg1_type": arg1_info["type"],
                    "arg2_id": arg2,
                    "arg2_text": arg2_info["text"],
                    "arg2_type": arg2_info["type"],
                }
                writer.writerow(row)
                relationship_count += 1

                if rel_type in UNDIRECTED_RELATIONS:
                    writer.writerow(
                        {
                            **row,
                            "arg1_id": row["arg2_id"],
                            "arg1_text": row["arg2_text"],
                            "arg1_type": row["arg2_type"],
                            "arg2_id": row["arg1_id"],
                            "arg2_text": row["arg1_text"],
                            "arg2_type": row["arg1_type"],
                        }
                    )
                    relationship_count += 1
    return relationship_count


def prepare_neo4j_csvs_streaming(
    input_path: str | Path,
    nodes_out: str | Path,
    rels_out: str | Path,
    total_relationships: int | None = None,
) -> None:
    nodes_out = Path(nodes_out)
    rels_out = Path(rels_out)
    nodes_out.parent.mkdir(parents=True, exist_ok=True)
    rels_out.parent.mkdir(parents=True, exist_ok=True)
    nodes_seen = set()

    with open(input_path, "r", encoding="utf-8") as fin, open(
        nodes_out, "w", newline="", encoding="utf-8"
    ) as fn, open(rels_out, "w", newline="", encoding="utf-8") as fr:
        reader = csv.reader(fin)
        n_writer = csv.writer(fn)
        r_writer = csv.writer(fr)

        n_writer.writerow([":ID", "name", "entity", "entity_type", ":LABEL"])
        r_writer.writerow([":START_ID", ":END_ID", ":TYPE", "pmid"])
        next(reader, None)

        rel_count = 0
        node_count = 0
        with tqdm(
            reader,
            total=total_relationships,
            desc="Build Neo4j CSVs",
            unit="relation",
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress:
            for row in progress:
                if len(row) < 8:
                    continue
                pmid, rel_type, mesh1, entity1, type1, mesh2, entity2, type2 = [
                    item.strip() for item in row[:8]
                ]
                if not mesh1 or not mesh2 or not rel_type:
                    continue
                if mesh1 not in nodes_seen:
                    n_writer.writerow([mesh1, mesh1, entity1, type1, "MESH"])
                    nodes_seen.add(mesh1)
                    node_count += 1
                if mesh2 not in nodes_seen:
                    n_writer.writerow([mesh2, mesh2, entity2, type2, "MESH"])
                    nodes_seen.add(mesh2)
                    node_count += 1
                r_writer.writerow([mesh1, mesh2, safe_rel_type(rel_type), pmid])
                rel_count += 1
                if rel_count % 10000 == 0:
                    progress.set_postfix(nodes=node_count, refresh=False)

    print(f"Done. Nodes: {node_count}, relationships: {rel_count}")


def normalize_entity_id(value) -> str:
    value = "" if value is None else str(value)
    return value.split(":", 1)[1] if value.startswith("MESH:") else value


def safe_rel_type(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value.upper())
