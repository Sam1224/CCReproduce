# Memento (toy reproduction)

- Paper: **Memento: Personalized RAG-Style Long-Retention Data Scaling for META Ads Recommendation** (arXiv:2605.24051, submitted 2026-05-22)
- Idea (high level): treat a user’s long engagement history as a *document corpus*; treat the current ad request as a *query*; retrieve relevant-yet-diverse past interactions (MMR) to mitigate long-context dilution / forgetting.

This folder implements a **toy but runnable** version of the core pipeline:

1. **Representation Memento**: retrieve `k` past interactions with MMR, aggregate their embeddings, and feed to a CTR/CVR predictor.
2. **Data Memento**: during training, retrieve similar past examples and replay them as auxiliary supervision (a simplified multi-pass training signal).

> Notes
>
>- The original paper focuses on production-scale infrastructure (temporal chunking, INT8 quantization, async serving). Here we keep the *interfaces* and *retrieval logic* aligned, but use a small synthetic dataset and a lightweight model.

## Quickstart

```bash
python -m pip install -r requirements.txt

# Baseline: LastN history aggregation
python train.py --mode lastn --epochs 5

# Representation Memento: history retrieval + aggregation
python train.py --mode memento_repr --epochs 5

# Data Memento: retrieval-based replay during training
python train.py --mode memento_data --epochs 5

# Evaluate a saved checkpoint
python test.py --ckpt runs/memento_repr.pt
```

## Files

- `data.py`: synthetic ads dataset with long history.
- `retrieval.py`: MMR retrieval (similarity + diversity).
- `model.py`: small CTR/CVR predictor.
- `train.py`: training for three modes.
- `test.py`: evaluation (AUC / logloss).
