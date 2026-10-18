# Project Overview

## Ablation Details

### Meaning of the `Candidate Mechanisms` Field

- `Constraints`: Constrains both entity types and relation types; this is the method proposed in this study.
- `Causality`: Constrains only relation types with an explicit direction of effect.
- `All`: Includes all two-hop paths.

### Meaning of the `variant` Field

- `full`: Runs the complete model.
- `oe`: When running `sort.py`, excludes opposing evidence. The ranking results are saved to `sort-oe.jsonl`.
- `cs`: When running `sort.py`, excludes the context score weight. The ranking results are saved to `sort-cs.jsonl`.
- `lw`: When running `sort.py`, excludes the literature count weight. The ranking results are saved to `sort-lw.jsonl`.

## Workflow

### Download Papers and Build the Knowledge Graph

Run:

```bash
python build_graph.py --config configs/fira.json
```

### Search for Candidate Mechanisms

Start Neo4j, then run:

```bash
python search.py --config configs/fira.json
```

### LLM-Based Filtering

This step classifies the direction of effect and evaluates contextual consistency.

```bash
python judge.py --config configs/fira.json
```

### Rank Intermediate Concepts

```bash
python sort.py --config configs/fira.json
```
