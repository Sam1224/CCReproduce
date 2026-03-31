# msa (toy reproduction)

Toy, runnable reproduction skeleton for:

- **MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens** (arXiv:2603.23516)

## What this reproduction captures

MSA’s headline goal is **lifetime-scale memory** with (near-)linear compute, by combining a **memory selector** (top-k) and a **sparse attention** backbone, plus practical inference optimizations.

This toy implementation focuses on the *mechanism-level loop*:

- A set of “documents” (memory slots) is embedded.
- A differentiable **selector** produces a soft distribution and a hard top-k subset.
- A small **cross-attention** module attends only to the selected memory.
- The model predicts an answer label for a synthetic query.

It is **not** a faithful 100M-token system reproduction; it is a clean reference for the selector + sparse cross-attention pattern.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-29/msa
python3 -m pip install -r requirements.txt
python3 train.py --epochs 5
python3 test.py
```

## Files

- `dataset.py`: synthetic memory/query classification task
- `model.py`: differentiable selector + sparse cross-attention
- `train.py`: training loop
- `test.py`: smoke test
