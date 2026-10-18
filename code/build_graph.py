import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_build_config
from graph_building import build_kg_csv, download_pubmed, extract_semrep, make_import_command

CONFIG_PATH = "configs/fira.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PubMed, SemRep, and Neo4j CSV files for CE-LBD.")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to the case config JSON.")
    args = parser.parse_args()

    config = load_build_config(args.config)
    step = config.get("build_step", "all")
    valid_steps = {"all", "download", "extract", "kg", "import-command"}
    if step not in valid_steps:
        raise ValueError(f"Invalid build step {step!r}; choose one of: {', '.join(sorted(valid_steps))}")

    if step in {"all", "download"}:
        print("\n[1/4] Download PubMed records")
        download_pubmed.run(config)
    if step in {"all", "kg"}:
        print("\n[2/4] Extract SemRep predications")
        extract_semrep.run(config)
        print("\n[3/4] Build knowledge graph CSV files")
        build_kg_csv.run(config)
        print("\n[4/4] Create Neo4j import command")
        make_import_command.run(config)

    if step == "all":
        print("\nGraph files are ready.")
        print("Next manual steps:")
        print("1. Stop Neo4j.")
        print(f"2. Run: {config['import_cmd_path']}")
        print(f"3. Set dbms.default_database={config['neo4j_database']} in neo4j.conf if needed.")
        print("4. Start Neo4j.")
        print(f"5. Run: python search.py --config {args.config}")


if __name__ == "__main__":
    main()
