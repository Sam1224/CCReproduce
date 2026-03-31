# MemoryCD (reproduction-oriented benchmark harness)

Reference paper: **MEMORYCD: Benchmarking Long-Context User Memory of LLM Agents for Lifelong Cross-Domain Personalization** (arXiv:2603.25973)

This folder implements a **standalone benchmark harness** that mirrors the key benchmark structure:

- User memories spanning **multiple domains** and long interaction histories
- 4 personalization tasks:
  - rating prediction (MAE/RMSE)
  - item ranking (NDCG@K)
  - review title summarization (ROUGE-L / BLEU-1)
  - review generation (ROUGE-L / BLEU-1)
- Two evaluation settings:
  - single-domain memory
  - cross-domain memory

Because the original paper evaluates long-context LLM agents, this reproduction focuses on:

- dataset/task formalization
- memory-method baselines (non-LLM heuristics)
- metric implementation

## Quickstart

```bash
cd ccreproduce_repo/2026-03-31/memorycd
python3 -m pip install -r requirements.txt
python3 run.py --users 200 --history 200
```

## Using real datasets (pseudocode)

The paper uses Amazon Reviews. Converting the full dataset requires external storage and the `datasets` library.

```text
# NOT IMPLEMENTED (requires large dataset download)
for user in amazon_reviews:
  build_long_context_memory(user)
  sample_tasks(user)
  export_jsonl(memorycd_format)
```
