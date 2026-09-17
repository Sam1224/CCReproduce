# CaseDrivenSearchAgents (Toy Reproduction)

Toy-but-runnable PyTorch reproduction for:

- **A Case-Driven Multi-Agent Framework for E-Commerce Search Relevance** (arXiv:2605.05991)

## What is reproduced

This implementation keeps the paper's core closed-loop system design:

- a **User Agent** that mines underestimation and mismatch bad cases;
- an **Annotator Agent** that relabels cases under evolving standards;
- an **Optimizer Agent** that turns bad cases into data augmentation and model updates;
- a lightweight **Global Memory** that records resolved cases and standard updates;
- a shared **all-in-one relevance model** with retrieval / coarse-rank / fine-score heads.

The toy task is four-level relevance classification for e-commerce query-product pairs: `irrelevant / weak / relevant / strong`. Evaluation reports accuracy, macro-F1, relevant precision/recall, and an online-like `win_rate` proxy.

## Files

- `data.py`: synthetic e-commerce relevance data and evaluation metrics.
- `model.py`: lightweight all-in-one relevance model.
- `train.py`: baseline training plus case-driven multi-agent optimization loop.
- `test.py`: checkpoint loading and baseline-vs-multi-agent comparison.

## Quickstart

```bash
python train.py
python test.py
```

## Expected outcome

The multi-agent system should outperform the single-pass baseline, reflecting the paper's main point: **search relevance improves when bad-case discovery, annotation, optimization, and memory are connected into a closed loop rather than handled as isolated manual steps**.
