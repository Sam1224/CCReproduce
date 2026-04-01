# RDPO + Sparse MoE (toy reproduction)

Paper: **Aligning Multimodal Sequential Recommendations via Robust Direct Preference Optimization with Sparse MoE** (arXiv:2603.29259)

This folder provides a minimal runnable scaffold:

- a toy multimodal sequential recommendation dataset generating (pos_next, neg_next) preference pairs
- a small Transformer encoder with a **Top-1 MoE** block
- a DPO-style pairwise loss (`softplus(-beta*(pos-neg))`)

## Run

```bash
python3 train.py --epochs 2
python3 test.py --ckpt ckpt.pt
```

## Not implemented

- The paper’s full robust negative sampling strategy and production-scale multimodal features.
- Explicit mixture-of-experts regularization and ablations.
