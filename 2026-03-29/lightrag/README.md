# lightrag (toy reproduction)

Toy, runnable reproduction skeleton for:

- **LightRAG: Simple and Fast Retrieval-Augmented Generation** (arXiv:2410.05779)

## What this reproduction captures

LightRAG’s key idea is to **inject graph structure** into indexing / retrieval:

- **Low-level retrieval** over text chunks (vector-like / lexical similarity)
- **High-level retrieval** over a **graph of entities and relations** to discover connected information
- **Incremental updates** so new knowledge can be integrated quickly

This toy implementation builds:

- a small entity co-occurrence graph from synthetic documents
- a lightweight bag-of-words retrieval over chunks
- a graph expansion retrieval (BFS from query entities)
- a simple fusion reranker

## Quickstart

```bash
cd ccreproduce_repo/2026-03-29/lightrag
python3 train.py
python3 test.py
```

## Files

- `kb.py`: document store + entity graph construction
- `retriever.py`: dual-level retrieval + fusion
- `train.py`: demo queries + incremental update
- `test.py`: smoke test
