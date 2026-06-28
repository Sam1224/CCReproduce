# EmbeddingModelV3 (Toy Reproduction)

Toy-but-runnable reproduction for:

- **Scaling Dense Retrieval with LLM-Annotated Training Data: Structured Mining and Progressive Curriculum for E-Commerce Sponsored Search** (arXiv:2606.23911)

## Goal

Provide a minimal, end-to-end pipeline that mirrors the *core recipe* of the paper:

- **Structured mining** from multi-channel retriever disagreement (lexical vs dense) to form easy/hard positives and hard negatives.
- **(Simulated) LLM annotation cascade** to produce 5-grade relevance labels (0–4) at scale.
- **Progressive curriculum** to train a two-tower dense retriever: BCE → in-batch MNR (InfoNCE) → triplet.

This implementation is *toy but runnable*: it uses a synthetic sponsored-search dataset and lightweight models, while keeping interfaces aligned with an industrial retrieval pipeline.

## Quickstart

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## What you should see

The script prints:

- Dataset stats + mining bucket sizes
- A small evaluation report on a held-out set (NDCG@10 / Recall@10)
- Improvements after curriculum stages (BCE → MNR → Triplet)

## Files

- `data.py`: synthetic sponsored-search queries/products + ground-truth 5-grade relevance.
- `mining.py`: multi-channel retrieval + disagreement mining into difficulty buckets.
- `annotator.py`: simulated model cascade + calibration to produce pseudo labels.
- `model.py`: tiny dual-encoder retriever.
- `train.py`: curriculum training loops (BCE / MNR / Triplet).
- `eval.py`: retrieval evaluation (NDCG@10, Recall@K).
- `run_pipeline.py`: orchestration entry.
