from __future__ import annotations

from pathlib import Path


def run(config: dict) -> Path:
    output_path = Path(config["import_cmd_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    neo4j_admin = Path(config["neo4j_home"]) / "bin" / "neo4j-admin.bat"
    script = f"""@echo off
REM Stop Neo4j before running this import command.
"{neo4j_admin}" import ^
  --database={config["neo4j_database"]} ^
  --nodes="{config["nodes_csv_path"]}" ^
  --relationships="{config["relationships_csv_path"]}" ^
  --delimiter=, ^
  --skip-bad-relationships=true ^
  --skip-duplicate-nodes=true ^
  --force=true
"""
    output_path.write_text(script, encoding="utf-8")
    print(f"Manual Neo4j import command written to: {output_path}")
    return output_path
