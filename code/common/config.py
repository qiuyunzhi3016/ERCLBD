import json
from common.mechanisms import candidate_mode
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_path_config(path_name: str, variant: str = "full") -> dict:
    path = PROJECT_ROOT / "configs" / f"{path_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Path config does not exist: {path}")
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)
    config["project_root"] = str(PROJECT_ROOT)
    config["path_name"] = config.get("path_name", path_name)
    config = apply_variant(config, variant)
    return normalize_paths(config)


def load_build_config(config_path: str) -> dict:
    path = resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Build config does not exist: {path}")

    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)

    config["project_root"] = str(PROJECT_ROOT)
    config["path_name"] = config.get("path_name", path.stem)
    config["variant"] = "full"
    config.setdefault("build_step", "all")
    return normalize_paths(config)


def load_search_config(config_path: str) -> dict:
    path = resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Search config does not exist: {path}")

    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)

    path_name = config.get("path_name", path.stem)
    config.setdefault("Candidate Mechanisms", "Constraints")
    candidate_mode(config)
    config["project_root"] = str(PROJECT_ROOT)
    config["path_name"] = path_name
    config["variant"] = "full"
    return normalize_paths(config)


def resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def apply_variant(config: dict, variant: str) -> dict:
    if variant in {"", "full", None}:
        config["variant"] = "full"
        return config

    if variant not in {"nobroad", "nocr"}:
        raise ValueError(
            f"Unsupported variant {variant!r}; choose one of: full, nobroad, nocr"
        )

    variant_config = dict(config)
    variant_config["path_name"] = f"{config['path_name']}-{variant}"
    variant_config["variant"] = variant
    variant_config["output_dir"] = str(PROJECT_ROOT / "output" / variant_config["path_name"])
    return variant_config


def normalize_paths(config: dict) -> dict:
    root = PROJECT_ROOT
    output_dir = Path(config.get("output_dir", root / "output" / config["path_name"]))
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    path_name = config["path_name"]
    config["output_dir"] = str(output_dir)
    config.setdefault("pubmed_dir", str(output_dir / "pubmed-data"))
    config.setdefault("extract_dir", str(output_dir / "pubmed-extract"))
    config.setdefault("kg_dir", str(output_dir / "pubmed-kg"))
    config.setdefault("paths_dir", str(output_dir / "paths"))
    config.setdefault("logs_dir", str(output_dir / "logs"))
    config.setdefault("pubmed_jsonl_path", str(Path(config["pubmed_dir"]) / "merged-output.jsonl"))
    config.setdefault("semrep_jsonl_path", str(Path(config["extract_dir"]) / f"{path_name}-semrep.jsonl"))
    config.setdefault("kg_csv_path", str(Path(config["kg_dir"]) / f"{path_name}-kg.csv"))
    config.setdefault("nodes_csv_path", str(Path(config["kg_dir"]) / f"{path_name}-nodes.csv"))
    config.setdefault("relationships_csv_path", str(Path(config["kg_dir"]) / f"{path_name}-relationships.csv"))
    config.setdefault("import_cmd_path", str(Path(config["kg_dir"]) / f"neo4j-import-{path_name}.cmd"))
    config.setdefault("cfs_path", str(Path(config["paths_dir"]) / "cfs.jsonl"))
    config.setdefault("crs_path", str(Path(config["paths_dir"]) / "crs.jsonl"))

    config.setdefault("bs_path", str(Path(config["paths_dir"]) / "bs.jsonl"))
    config.setdefault("bfs_path", str(Path(config["paths_dir"]) / "bfs.jsonl"))
    config.setdefault("brs_path", str(Path(config["paths_dir"]) / "brs.jsonl"))
    config.setdefault("sort_output_path", str(Path(config["paths_dir"]) / "sort.jsonl"))
    config.setdefault("confidence_cache_path", str(Path(config["paths_dir"]) / "sort-confidence-cache.jsonl"))
    create_output_dirs(config)
    return config


def create_output_dirs(config: dict) -> None:
    directory_keys = (
        "output_dir",
        "pubmed_dir",
        "extract_dir",
        "kg_dir",
    )
    for key in directory_keys:
        Path(config[key]).mkdir(parents=True, exist_ok=True)
