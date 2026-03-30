# DIET (Diet Your LLM) — toy reproduction

This folder is a **runnable reproduction skeleton** of the key algorithmic idea in:

- **Diet Your LLM: Dimension-wise Global Pruning of LLMs via Merging Task-specific Importance Score** (arXiv:2603.23985)

## What this reproduction covers

The original paper proposes **training-free structured pruning**:

1. For each task, run a small number of samples (paper: 100) through the model.
2. Compute **dimension-wise activation magnitude** as an importance score.
3. Convert each task’s importance into a **task-specific keep/prune mask**.
4. Merge task masks via **majority voting** to produce a single **global** pruning mask.

In a real LLM, “dimension-wise structured pruning” removes entire hidden dimensions from weight matrices. For a lightweight toy reproduction, we apply the **same global mask** as a multiplicative gating vector on the hidden states (equivalent to pruning for forward passes), keeping the code small and runnable.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-25/diet_your_llm/DIET
python3 -m pip install -r requirements.txt
python3 train.py
python3 test.py
```

## Files

- `dataset.py`: 3 toy tasks (add/parity/compare) to mimic multi-task profiling
- `model.py`: tiny causal Transformer LM + per-dimension mask
- `diet.py`: activation profiling + majority vote mask construction
- `train.py`: trains a small base model on toy tasks (not part of DIET, just for having a usable base)
- `test.py`: evaluates baseline vs. DIET-pruned (masked) model
