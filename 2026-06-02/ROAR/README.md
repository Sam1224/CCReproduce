# ROAR (toy reproduction)

- Paper: **Semantic Retrieval for Product Search in E-Commerce** (arXiv:2606.01504, Flipkart)
- Core idea:
  - Use a **Siamese dual-encoder** for large-scale product retrieval.
  - Stage 1: **contrastive learning** with a *false-negative margin mask* (avoid penalizing near-duplicate / variant products that appear as in-batch negatives).
  - Stage 2: **ROAR (Relative Odds Alignment for Retrieval)**: optimize *graded* relevance ordering (Exact > Substitute > Complementary > Irrelevant) via consecutive **odds-ratio** margins.

This folder implements a **toy but runnable** version of the *training objectives and batching interfaces*.

> Notes
>
>- The paper uses Qwen3-Embedding-4B + LoRA + Matryoshka representations and proprietary data. Here we focus on reproducing the *loss functions* and the *two-stage training pipeline*, using a small synthetic ESCI-like dataset and a lightweight encoder.

## Quickstart

```bash
python -m pip install -r requirements.txt

# Stage 1: contrastive + false-negative margin masking (delta=0.1)
python train_stage1.py --epochs 5

# Stage 2: add ROAR alignment loss
python train_stage2.py --stage1 runs/stage1.pt --epochs 3
```

## Files

- `data.py`: synthetic ESCI-like dataset builder.
- `model.py`: simple shared dual-encoder + tokenizer.
- `losses.py`: false-negative margin masking + ROAR alignment loss.
- `train_stage1.py`: stage-1 training.
- `train_stage2.py`: stage-2 fine-tuning + quick evaluation.
- `evaluate.py`: MAP@8 / NDCG@8 / AUC on grouped candidates.
