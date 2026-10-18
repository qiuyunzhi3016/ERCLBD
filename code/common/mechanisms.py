"""Shared candidate modes and on-disk schema."""
from pathlib import Path

MODES = {"All": "as", "Constraints": "bs", "Causality": "cs"}
PATH_FIELDS = ["start_node", "start_text", "mid_node", "mid_text", "mid_entity_type",
               "end_node", "end_text", "rel1", "rel2", "pmid1", "pmid2",
               "pmid1_evidence", "pmid2_evidence"]

def candidate_mode(config):
    mode = config.get("Candidate Mechanisms", "Constraints")
    if mode not in MODES:
        raise ValueError(f"Invalid Candidate Mechanisms: {mode!r}; choose {list(MODES)}")
    return mode

def candidate_file(config, classified=False):
    stem = MODES[candidate_mode(config)]
    return Path(config["paths_dir"]) / (stem + ("-filter" if classified else "") + ".jsonl")

def project_path(path, classified=False):
    return {key: path.get(key, "") for key in PATH_FIELDS + (["label", "score"] if classified else [])}
