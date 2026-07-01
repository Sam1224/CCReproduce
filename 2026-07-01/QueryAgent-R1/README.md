# QueryAgent-R1 — Toy Reproduction

Paper: QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation  
arXiv: https://arxiv.org/abs/2606.05671

This folder implements a compact PyTorch reproduction of the paper's core pipeline: memory abstraction over long user behavior, candidate query scoring, product-search-grounded consistency evaluation, and query/item/consistency metrics.

Run:

```bash
pip install -r requirements.txt
python run_pipeline.py
```

The toy implementation replaces the production inventory and Qwen policy model with synthetic products and a small neural ranker, while preserving the method logic of retrieval-grounded query generation and downstream item-hit validation.
