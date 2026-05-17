# DualPathMod — Dual-Path Content Moderation with MLLM Distillation

Faithful PyTorch reproduction from:

> **Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching**  
> arXiv: 2512.03553 | ACM SIGKDD 2026

## Structure

```
DualPathMod/
├── README.md
├── model/
│   ├── dual_path.py            # Main dual-path moderation model
│   ├── mllm_distill.py         # MLLM knowledge distillation
│   └── similarity_store.py     # Violation case store + kNN retrieval
├── data/
│   └── livestream_dataset.py   # Toy livestream dataset
├── train.py                    # Training: distillation + classification
└── eval.py                     # Eval: recall@precision, AUC
```

## Quick Start

```bash
pip install torch faiss-cpu numpy scikit-learn

# Train both paths with MLLM distillation
python train.py --epochs 2 --batch_size 4

# Evaluate recall@80% precision
python eval.py --checkpoint ./checkpoints/dualpath.pt
```

## Key Metrics (Paper)

| Path | Recall @ 80% Precision | Production A/B |
|------|----------------------|----------------|
| Supervised Classification | 67% | — |
| MLLM-Boosted Similarity | 76% | 6-8% ↓ unwanted views |
