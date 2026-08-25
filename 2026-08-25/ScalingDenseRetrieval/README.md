# ScalingDenseRetrieval (Toy Reproduction)

Toy-but-runnable reproduction for:

- **Scaling Dense Retrieval with LLM-Annotated Training Data: Structured Mining and Progressive Curriculum for E-Commerce Sponsored Search** (arXiv:2606.23911)

## Goal

This folder reproduces the paper's **core training-data pipeline** in a minimal offline setting rather than Walmart's production stack.

It keeps four central ideas:

1. **Multi-channel disagreement mining**: lexical / taxonomy / behavior retrieval channels disagree on what is relevant.
2. **LLM-style cascade labeling**: a three-stage synthetic judge produces graded relevance labels.
3. **Five difficulty levels**: easy consensus positives, single-channel hard positives, high-rank hard negatives, and ambiguous cases.
4. **Progressive curriculum**: train a two-tower dense retriever from easy cases to hard cases.

## What is implemented

- `data.py`: synthetic e-commerce sponsored-search generator with channel disagreement, cascade grading, and difficulty buckets.
- `model.py`: lightweight bi-encoder dense retriever.
- `train.py`: three-stage curriculum training with pairwise ranking loss.
- `test.py`: offline evaluation with `NDCG@10`, `Recall@10`, `Embarrassing@1`, and head/tail breakdown.

## What is simplified

- No proprietary Walmart traffic logs, no 240M-scale data, and no real LLM annotation costs.
- The LLM cascade is simulated by progressively stronger heuristic judges.
- Retrieval is brute-force over toy candidates instead of ANN + production serving.

## Quickstart

```bash
python train.py --generate-data
python test.py --checkpoint checkpoints/scaling_dense_retrieval.pt
```

## Expected outcome

You should observe that the curriculum-trained retriever improves `NDCG@10` while reducing `Embarrassing@1`, especially on tail queries where lexical-only systems are brittle.
