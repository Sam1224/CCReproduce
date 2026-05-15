# CQ-SID + EG-GRPO (Toy Reproduction)

This folder provides a **toy** PyTorch reproduction for the paper:

> **Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL** (arXiv:2605.14434)

It implements a simplified but end-to-end pipeline aligned with the paper’s core ideas:

- **CQ-SID**: build **semantic cluster IDs** for items using a **3-level Residual Vector Quantizer (RQ-VAE style)**.
- **Generative retrieval**: a small seq2seq Transformer maps *(category + query + (optional) user preference)* to **SID token sequences**.
- **EG-GRPO**: a simplified **group-relative policy optimization** loop; each group injects the **ground-truth SID sequence** as an expert anchor to stabilize learning under sparse rewards.

## Files

- `dataset.py`: toy e-commerce dataset generation (items, queries, users) + tokenization
- `model.py`: RQ-VAE (EMA codebooks) + seq2seq Transformer + EG-GRPO utilities
- `train.py`: stage-wise training pipeline (RQ-VAE -> SFT -> EG-GRPO)
- `test.py`: evaluate hit@K on the toy set

## Quickstart

```bash
pip install -r requirements.txt
python train.py
python test.py
```

## Notes / Differences from the paper

- This is **not** a scale reproduction (no real logs, no online A/B).
- The paper’s “category-aware level-1 codebook” is reproduced by maintaining **per-category** codebooks for the first quantization level.
- The paper describes a 4-stage progressive training. This toy code keeps the spirit (item->SID training, then query->SID SFT, then RL refinement) but uses smaller models and simplified objectives.
