# DSIRM (toy reproduction)

- Paper: **DSIRM: Learning Query-Bridged Discrete Semantic Identifiers for E-commerce Relevance Modeling** (arXiv:2606.04374, Taobao & Tmall Group)
- Core idea:
  - Learn **hierarchical discrete Semantic IDs (SIDs)** for items via **Residual Quantization (RQ)**, but *inject query–item interaction supervision* so the partitions become *relevance-aware*.
  - Learn a **query → SID** predictor (paper uses a generative LLM) to disambiguate tail / short queries.
  - Use **hierarchical prefix matching** between query-SID and item-SID as additional discrete relevance features that complement dense embeddings.

This folder implements a **toy but runnable** end-to-end pipeline focusing on the *method logic* and *interfaces*:

1. Stage-1: train dual-encoder + **contrastive RQ quantizer** (query-bridged InfoNCE) to produce item SIDs.
2. Stage-2: train a lightweight **query-SID predictor** (simplified from the paper’s generative LLM component).
3. Stage-3: train a small **ranker** that combines dense similarity + SID prefix-match features.

> Notes
>
>- The original paper is trained / evaluated on large proprietary Tmall traffic data and uses an LLM for query SID generation. In this toy reproduction we:
>  - use a synthetic attribute-heavy product corpus (colors/materials/sizes/styles) to simulate “high lexical overlap but fine-grained attribute distinctions”.
>  - keep the hierarchical SID + prefix-match mechanism consistent.
>  - implement a small query→SID model (multi-head predictor) instead of fine-tuning a large LLM.

## Quickstart

```bash
python -m pip install -r requirements.txt

# Stage 1: learn relevance-aware item SIDs
python train_stage1_item_sids.py --epochs 8

# Stage 2: train query -> SID predictor
python train_stage2_query_sids.py --epochs 5 --stage1 runs/stage1.pt

# Stage 3: train ranker using dense + SID prefix features
python train_stage3_ranker.py --epochs 5 --stage1 runs/stage1.pt --stage2 runs/stage2.pt

# Evaluate (AUC + NDCG@10 on grouped candidates)
python evaluate.py --stage1 runs/stage1.pt --stage2 runs/stage2.pt --stage3 runs/stage3.pt
```

## Files

- `data.py`: synthetic e-commerce query/product dataset builder.
- `model.py`: tiny tokenizer + shared dual-encoder.
- `rqvae.py`: residual vector quantizer (RQ-VAE-style) producing hierarchical item SIDs.
- `sid.py`: prefix match features.
- `train_stage1_item_sids.py`: stage-1 training (contrastive + quantization losses).
- `train_stage2_query_sids.py`: stage-2 training (query → SID prediction).
- `train_stage3_ranker.py`: stage-3 ranking.
- `evaluate.py`: AUC + NDCG@10 on grouped candidates.
