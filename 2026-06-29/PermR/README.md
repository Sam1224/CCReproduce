# PermR — Permutation-based Constrained Reranking (Toy Reproduction)

- Paper: **Fast and Feasible: Permutation-based Constrained Reranking for Revenue Maximization**
- arXiv: https://arxiv.org/abs/2606.28059

## What is reproduced

The paper formulates revenue-aware reranking as an ILP: maximize revenue subject to per-query constraints (e.g., relevance / fraud safety) relative to the production ranking, then proposes **PermR**, a lightweight neighbor-swap approximation algorithm for production latency.

This folder provides a **toy, self-contained reproduction** that covers the key idea:

- A baseline ranking produced by a learned relevance scorer.
- Constraints measured on **NDCG@K** (relevance) and **AvgRisk@K** (risk proxy).
- Objective measured on a **position-discounted revenue** over the full ranked list (to mimic exposure).
- An **exact ILP solution** implemented by enumerating all permutations (small N only).
- A **PermR-like neighbor-swap** optimizer that (1) repairs constraint violations, then (2) improves revenue under constraints.

> Note: This is a faithful algorithmic reproduction on a synthetic dataset. The paper’s industrial results are on a large classifieds platform; we cannot reproduce those private logs here.

## Files

- `data.py`: synthetic query → candidate-item generator
- `model.py`: tiny PyTorch multi-task scorer (relevance + revenue heads)
- `train.py`: trains the scorer on the toy data
- `rerank.py`: baseline ranking, exact ILP (enumeration), and PermR-like algorithm
- `eval.py`: evaluates baseline vs PermR vs ILP on held-out synthetic queries
- `run_pipeline.py`: runs `train.py` then `eval.py`

## Quickstart

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Expected output (numbers vary by random seed), printed:

- Revenue@K for baseline / PermR / ILP(opt)
- PermR’s revenue-gain ratio vs ILP
- NDCG@K and AvgRisk@K for constraint sanity

## How to map to the paper

- **ILP**: `ilp_exact_by_enumeration(...)` corresponds to the paper’s ILP objective+constraints, but solved exactly by brute force for small N.
- **PermR**: `permr_rerank(...)` follows the paper’s high-level loop of swapping neighboring items to either (a) repair violated constraints or (b) increase the objective while keeping constraints satisfied.

## Limitations

- Exact ILP is only feasible for small candidate lists (default `N=8`).
- Risk/relevance/revenue are synthetic proxies; the goal is to reproduce the *optimization structure* rather than private business distributions.
