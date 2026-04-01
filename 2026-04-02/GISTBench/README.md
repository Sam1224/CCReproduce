# GISTBench (Toy Reproduction)

Paper: **GISTBench: A Framework for Evaluating User Interest Extraction in LLMs** (arXiv:2603.29112)

This folder provides a **toy, runnable reproduction** of the benchmark-style pipeline:

- synthetic user interaction histories (UIH) with mixed signals
- a learnable baseline interest extractor (PyTorch)
- proxy metrics for:
  - **Interest Groundedness (IG)**: precision/recall/F1 against known synthetic ground truth
  - **Interest Specificity (IS)**: taxonomy-based specificity score
- a small demonstration of correlation between the evaluation score and a simulated “user survey” signal

## Quickstart

```bash
pip install torch

# Train a baseline interest extractor
python3 train.py --out ckpt.pt

# Evaluate IG / IS and correlation
python3 test.py --ckpt ckpt.pt
```

## Notes

- The original paper evaluates **LLMs** and introduces verification-style metrics without explicit ground-truth labels.
  In this toy reproduction we use synthetic construction so that ground truth is available, and implement the same *shape*
  of evaluation logic in a fully runnable form.

