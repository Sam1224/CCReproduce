# CaseDrivenMultiAgentRelevance (Toy Reproduction)

Toy-but-runnable reproduction for:

- **A Case-Driven Multi-Agent Framework for E-Commerce Search Relevance** (arXiv:2605.05991)

## Goal

Replicate the *core loop* in a minimal setting:

- **User Agent** discovers bad cases (model underestimation / mismatch).
- **Annotator Agent** labels cases under evolving natural-language-like standards (here: explicit rules + synonym tables).
- **Optimizer Agent** repairs the system via data augmentation + multi-task retraining.
- A lightweight **Global Memory** tracks resolved cases and rule updates.

This repo does **not** reproduce ByteDance production infrastructure, online SBS A/B, or proprietary datasets.
Instead it provides an end-to-end runnable pipeline (toy data / model / agents / train / eval) that matches the paper's system logic.

## Quickstart

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## What you should see

- Baseline accuracy improves after each agent iteration.
- The log prints an *online-like* "win-rate gain" proxy computed from a held-out set.

## Files

- `data.py`: synthetic e-commerce query–product pairs + labels.
- `model.py`: a tiny "All-In-One" relevance model (retrieval + coarse rank + fine rank heads).
- `agents.py`: user/annotator/optimizer agents + global memory.
- `train.py`: training loop.
- `eval.py`: evaluation metrics.
- `run_pipeline.py`: orchestrates the multi-agent loop.
