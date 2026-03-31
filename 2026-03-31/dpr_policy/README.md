# Open-Domain Safety Policy Construction (DPR-style reproduction)

Reference paper: **Open-Domain Safety Policy Construction** (EACL Findings)

This folder implements a practical, runnable pipeline for constructing a safety policy from open-domain sources:

- ingest a document corpus (local text files)
- retrieve relevant passages for each policy topic via TF-IDF
- synthesize a policy document template
- run a lightweight “coverage” evaluation by generating synthetic queries and checking if policy sections provide matching constraints

This is **not** a full LLM-based policy writer; instead it provides the core retrieval + structuring logic and a clean interface where an LLM could be plugged in.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-31/dpr_policy
python3 -m pip install -r requirements.txt
python3 run.py
```

## Pseudocode (LLM policy writer)

```text
# NOT IMPLEMENTED
for topic in topics:
  ctx = retrieve_passages(topic)
  section = LLM("Write policy section", ctx)
  policy.add(section)
```
