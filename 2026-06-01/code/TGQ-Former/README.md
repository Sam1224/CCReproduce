# TGQ-Former — Code Reproduction

**Paper:** Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation  
**arXiv:** https://arxiv.org/abs/2605.17366  
**Authors:** Yufei Guo, Jing Ma, Tianlu Zhang (Tsinghua); Shijie Yang, Yanlong Zang, Weijie Ding, Pinghua Gong (JD.COM); Jungong Han (Tsinghua)

## Architecture

```
Product Image                  Product Metadata (Title, Category, Attributes)
     │                                         │
Vision Encoder (frozen)               Text Encoder
     │                                         │
Visual Tokens (V)              Metadata Embedding (M)
     │                                         │
     ├────────── Hybrid-Query Connector ────────┤
     │                                         │
     │  ┌─────────────────────────────────┐    │
     │  │  Metadata-Anchored Query        │    │
     │  │  (Cross-attn: Q→V, guided by M) │    │
     │  │            +                    │    │
     │  │  Exploratory Query              │    │
     │  │  (Self-attn: Q→V, free)         │    │
     │  └─────────────────────────────────┘    │
     │                   │                     │
     │    Reliability-Aware Dual-Gated          │
     │    Vector Modulation                    │
     │    (adaptive fusion of two streams)     │
     │                   │                     │
     └───────────── Visual Embedding ──────────┘
                         │
              Aligned with Text Embedding
                         │
               I2I Retrieval (cosine similarity)
```

## Files

- `model.py` — TGQ-Former model (full architecture)
- `dataset.py` — Toy e-commerce product dataset
- `train.py` — Contrastive training script
- `evaluate.py` — I2I retrieval evaluation (Hit Rate @ K)

## Quick Start

```bash
pip install torch transformers numpy
python train.py --epochs 20 --batch_size 64
python evaluate.py --checkpoint checkpoints/best.pt
```
