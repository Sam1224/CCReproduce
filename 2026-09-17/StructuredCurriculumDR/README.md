# StructuredCurriculumDR (Toy Reproduction)

Toy-but-runnable PyTorch reproduction for:

- **Scaling Dense Retrieval with LLM-Annotated Training Data: Structured Mining and Progressive Curriculum for E-Commerce Sponsored Search** (arXiv:2606.23911)

## What is reproduced

This folder keeps the paper's core industrial recipe in a compact form:

- **structured mining** from lexical-vs-dense disagreement to build easy positives, hard positives, hard negatives, and medium pairs;
- a simulated **three-model annotation cascade** that generates graded pseudo labels with calibrated confidence;
- **progressive curriculum training** for a two-tower retriever: BCE → in-batch multi-negative ranking → triplet refinement;
- comparison against a weaker **click-like baseline** trained only on easy random negatives.

The toy dataset is sponsored-search flavored: query intent includes brand/category plus optional flavor, diet, and size constraints. Evaluation reports `NDCG@10`, `Recall@10`, tail-query metrics, and `embarrassing@1` (top-1 grade-0 retrieval rate).

## Files

- `data.py`: synthetic catalog/queries, structured mining, annotation cascade, curriculum builders, and evaluation.
- `model.py`: lightweight dual-encoder retriever.
- `train.py`: baseline training and curriculum training orchestration.
- `test.py`: checkpoint loading and baseline-vs-curriculum comparison.

## Quickstart

```bash
python train.py
python test.py
```

## Expected outcome

You should see the curriculum model outperform the click-like baseline on ranking quality and reduce embarrassing retrievals, matching the paper's qualitative story that **LLM-annotated supervision + difficulty-aware curriculum** is stronger than naive click-only training.
