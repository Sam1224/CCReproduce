# MOON (toy reproduction)

Toy reproduction skeleton for:

- **MOON: Generative MLLM-based Multimodal Representation Learning for E-commerce Product Understanding** (arXiv:2508.11999)

## Idea captured

This toy repo captures the *core engineering ideas* without reproducing the full MLLM scale:

- **Multi-image → one title**: each product has multiple images and one textual title.
- **Core-region vs full image**: we simulate a "core crop" view to suppress background noise.
- **Guided MoE for aspects**: category tokens and attribute tokens are routed to different FFN experts.
- **Contrastive representation learning**: train embeddings with in-batch negatives.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-31/moon
python3 -m pip install -r requirements.txt
python3 train.py --epochs 10
python3 test.py
```

## Files

- `dataset.py`: synthetic e-commerce products with category/attribute tokens, multiple images, and purchase-behavior-like pairs.
- `model.py`: a small multimodal encoder with a guided MoE text FFN.
- `train.py` / `test.py`: contrastive training and Recall@K evaluation.
