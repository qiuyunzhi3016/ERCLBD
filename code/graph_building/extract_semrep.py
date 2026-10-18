from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from common.io import read_jsonl, write_jsonl
from common.progress import tqdm


def run(config: dict) -> Path:
    output_path = Path(config["semrep_jsonl_path"])
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"SemRep JSONL already exists: {output_path}")
        return output_path

    pubmed_rows = read_jsonl(config["pubmed_jsonl_path"], required=True)
    pmids = [str(row.get("PMID", "")).strip() for row in pubmed_rows if row.get("PMID")]
    meta = {
        str(row.get("PMID", "")).strip(): {
            "title": row.get("Title", ""),
            "abstract": row.get("Abstract", ""),
        }
        for row in pubmed_rows
    }
    predications = query_pmids_from_db(config["semmed_sqlite"], pmids)
    rows = build_semrep_rows(predications, meta)
    write_jsonl(output_path, rows)
    print(f"Saved {len(rows)} SemRep records to {output_path}")
    return output_path


def query_pmids_from_db(db_path: str, pmid_list: list[str]) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    results = []
    batch_size = 1000
    batch_starts = range(0, len(pmid_list), batch_size)
    for i in tqdm(
        batch_starts,
        total=(len(pmid_list) + batch_size - 1) // batch_size,
        desc="Query SemMedDB",
        unit="batch",
        dynamic_ncols=True,
    ):
        batch = pmid_list[i : i + batch_size]
        placeholders = ",".join(["?"] * len(batch))
        query = f"SELECT * FROM predication WHERE PMID IN ({placeholders})"
        results.append(pd.read_sql_query(query, conn, params=batch))
    conn.close()
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def build_semrep_rows(df: pd.DataFrame, meta: dict[str, dict]) -> list[dict]:
    rows = []
    grouped = df.groupby("PMID")
    for pmid, group in tqdm(
        grouped,
        total=df["PMID"].nunique() if "PMID" in df else 0,
        desc="Build SemRep records",
        unit="record",
        unit_scale=True,
        dynamic_ncols=True,
    ):
        pmid = str(pmid)
        entities = {}
        relations = []
        for _, row in group.iterrows():
            subj = str(row.get("SUBJECT_CUI", "")).strip()
            obj = str(row.get("OBJECT_CUI", "")).strip()
            if subj:
                entities[subj] = {
                    "id": subj,
                    "text": row.get("SUBJECT_NAME", ""),
                    "type": row.get("SUBJECT_SEMTYPE", ""),
                }
            if obj:
                entities[obj] = {
                    "id": obj,
                    "text": row.get("OBJECT_NAME", ""),
                    "type": row.get("OBJECT_SEMTYPE", ""),
                }
            if subj and obj:
                relations.append({"arg1": subj, "arg2": obj, "type": row.get("PREDICATE", "")})
        doc = meta.get(pmid, {})
        rows.append(
            {
                "pmid": pmid,
                "title": doc.get("title", ""),
                "abstract": doc.get("abstract", ""),
                "entities": list(entities.values()),
                "relations": relations,
            }
        )
    return rows
