# Xuanwu (toy reproduction)

Paper: **Xuanwu: Evolving General Multimodal Models into an Industrial-Grade Foundation for Content Ecosystems** (arXiv:2603.29211)

This folder provides a **minimal runnable PyTorch pipeline** that mirrors the typical structure of an industrial multimodal moderation model (text + image feature fusion), but **does not attempt** to reproduce the full foundation-model training system described in the paper.

## What is implemented

- A toy multimodal dataset interface (`dataset.py`) producing `image_feat`, `token_ids`, and a binary moderation label.
- A small multimodal fusion classifier (`model.py`).
- `train.py` / `test.py` scripts.

## How to run

```bash
python3 train.py --epochs 3
python3 test.py --ckpt ckpt.pt
```

## Not implemented

- Real data collection/curation loops, OCR robustness, large-scale pretraining, and industrial evaluation suites described in the paper.
