# RoDPO (Toy Reproduction)

This folder contains a small, runnable PyTorch reproduction of the key idea in **RoDPO** (arXiv:2603.29259): using **stochastic Top-K negative sampling** to make DPO-style preference learning more robust under implicit feedback (false negatives).

This is **not** an official implementation, and uses a **toy sequential recommendation dataset** generated locally.

## What is implemented

- A simple GRU-based sequential recommender (scores next-item candidates).
- A two-stage training pipeline:
  - **Stage 1 (CE warmup)**: next-item cross-entropy training to get a stable reference policy.
  - **Stage 2 (DPO)**: preference optimization with RoDPO's **stochastic Top-K negative sampling**.
- An optional lightweight **SparseMoE** encoder module (as a small stand-in for the paper's sparse MoE idea).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) create toy data
python3 data/generate_toy.py --out data/toy.json --num-users 200 --num-items 1000 --seq-len 30

# 2) warmup (CE)
python3 train.py --data data/toy.json --out-dir runs/ce --epochs 3

# 3) RoDPO fine-tune (stochastic top-k negatives)
python3 train.py --data data/toy.json --out-dir runs/rodpo --epochs 3 --stage dpo --init-from runs/ce/model.pt --ref-from runs/ce/model.pt --neg-sampling topk --topk 50

# 4) evaluate
python3 eval.py --data data/toy.json --ckpt runs/rodpo/model.pt
```

## Key RoDPO detail reproduced

Given a user state \(s\) and the ground-truth next item \(i^+\), RoDPO samples a negative \(i^-\) **stochastically** from the model's current Top-K predicted items (excluding \(i^+\)) rather than always picking the hardest negative. This reduces repeated false-negative suppression while keeping negatives informative.

## Reference

- Paper: https://arxiv.org/abs/2603.29259
