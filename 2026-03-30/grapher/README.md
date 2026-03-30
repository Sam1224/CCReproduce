# GraphER (toy reproduction)

Toy reproduction skeleton for:

- **GraphER: An Efficient Graph-Based Enrichment and Reranking Method for Retrieval-Augmented Generation** (arXiv:2603.24925)

## Idea captured

We simulate a RAG reranking setting where:

- A retriever provides candidate passages.
- A **graph enrichment** step builds a small passage graph (edges by token overlap) and propagates scores.
- A reranker predicts final relevance using both passage features and graph-propagated neighborhood signals.

This is a lightweight stand-in for “graph-based enrichment + reranking”.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-30/grapher
python3 -m pip install -r requirements.txt
python3 train.py --epochs 5
python3 test.py
```
