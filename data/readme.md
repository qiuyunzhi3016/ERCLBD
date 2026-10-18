# Data Description

## Original Data

The data is from [SemRep Relation Classification](https://github.com/Michelle-Mings/SemRep_RelationClassification).

`datas\original\all.jsonl`

## Data Filtered by the 12 Specified Relation Types

Run:

```bash
python filter.py
```

`datas\filter\all_12_predicates.jsonl`

## Data Annotated by Annotator 1

`datas\labels\A1.jsonl`

## Data Annotated by Annotator 2

`datas\labels\A2.jsonl`

## Cases Re-annotated by Annotator 3 Due to Disagreements Between the First Two Annotators

`datas\labels\A3.jsonl`

## Stratified Test Set

Run:

```bash
python split.py
```

`datas\test\365.jsonl`

## LLM-Based Direction-of-Effect Classification

Run:

```bash
python llm.py
```

## Performance Evaluation

Run:

```bash
python evaluate.py
```
