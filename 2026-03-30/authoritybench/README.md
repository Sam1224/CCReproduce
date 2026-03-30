# AuthorityBench (toy reproduction)

Toy reproduction skeleton for:

- **AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation** (arXiv:2603.25092)

## Idea captured

We simulate “authority perception” in retrieval by:

- Each query has multiple candidate sources with an **authority score** and a **content relevance score**.
- A model must pick which sources to trust.
- Metric evaluates whether the chosen evidence comes from high-authority sources while remaining relevant.

This approximates the benchmark’s core question: do LLM/RAG systems prefer authoritative evidence?

## Quickstart

```bash
cd ccreproduce_repo/2026-03-30/authoritybench
python3 -m pip install -r requirements.txt
python3 train.py --epochs 5
python3 test.py
```
