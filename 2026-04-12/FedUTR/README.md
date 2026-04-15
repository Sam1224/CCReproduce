# FedUTR (Toy Reproduction)

Paper: **FedUTR: Federated Recommendation with Augmented Universal Textual Representation for Sparse Interaction Scenarios** (arXiv:2604.07351)

This folder provides a *toy but runnable* PyTorch reproduction of the main idea:
- In sparse federated recommendation, **ID-only item embeddings** are brittle.
- Augment with a **universal textual representation** (URM) and fuse it with interaction-driven representations (CIFM).
- Keep **client personalization** via a lightweight local adaptation (LAM-style) component that stays on-device.

## What is implemented
- A minimal federated training loop (FedAvg over shared parameters).
- A FedUTR model with three components:
  - **URM**: token-embedding + mean pooling → linear projection.
  - **CIFM**: a gated fusion of ID item embedding and universal text embedding.
  - **LAM (simplified)**: per-client user bias vector (kept local, not aggregated).
- Baseline: ID-only federated CF (FCF-style).
- Metrics: HR@10 and NDCG@10.

## What is simplified
- The paper extracts text embeddings from off-the-shelf foundation models (e.g., BERT). Here we train a tiny token embedding from scratch on toy vocab.
- The paper has a more detailed CIFM/LAM and a sparsity-aware variant (FedUTR-SAR). We implement a simplified gating fusion and local bias.

## Run
```bash
pip install -r requirements.txt
python3 train.py --rounds 30 --clients 20 --sparsity 0.98
python3 test.py  --checkpoint checkpoints/global.pt
```

You should observe that under high sparsity, the FedUTR variant is more stable than the ID-only baseline.
