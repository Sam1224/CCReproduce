# UNIVID — Code Reproduction

**Paper:** UNIVID: Unified Vision-Language Model for Video Moderation  
**ArXiv:** https://arxiv.org/abs/2606.05748  
**Authors:** Kejuan Yang, Yizhuo Zhang, Mingyuan Du, Yue Zhang, Dixin Zheng, Kaili Zhao, Yang Xiao, Hanzhong Liang, Kenan Xiao (ByteDance)

---

## Overview

UNIVID is ByteDance's unified VLM for industrial-scale video content moderation. It replaces 1,000+ policy-specific classifiers with a single policy-aware caption generator, achieving **-42.7% violation leakage** and **-37.0% overkill rate** in production deployment.

### Architecture

```
Video Frames → [Vision Encoder] → Visual Features
                                         ↓
Policy Context → [Policy Conditioner] → Fused Representation
                                         ↓
              [UNIVID Backbone (VLM)] → Policy-Aware Caption
                                         ↓
          ┌──────────────────────────────────────────┐
          │           Three-Stage Pipeline            │
          │  Stage A: Risk Filter (caption → rough score) │
          │  Stage B: UNIVID-Lite (fast decision)    │
          │         + UNIVID-RAG (leakage recall)    │
          │  Stage C: Trend Governance (embedding clusters) │
          └──────────────────────────────────────────┘
```

### Key Equations

- **Caption Loss (Eq. 1):** `L_cap = -(1/T) Σ_t log P(c_t | c_{<t}, proj(V), embed(π))`
- **Violation Loss (Eq. 2):** `L_vio = BCE(σ(f(pooled)), y_violation)`
- **Total Loss (Eq. 3):** `L = 0.5 * L_cap + 0.5 * L_vio + 0.1 * L_policy`
- **Trend Score:** `trend_score = max_k cos(e, μ_k)` where μ_k are cluster centroids

---

## Files

| File | Description |
|------|-------------|
| `data.py` | Synthetic video moderation dataset with policy-aware captions |
| `model.py` | Full UNIVID system: Backbone + Lite + RAG + Trend head |
| `train.py` | 4-phase training pipeline (backbone → RAG index → Lite → Trend init) |
| `evaluate.py` | Three-stage inference evaluation with leakage/overkill metrics |
| `requirements.txt` | Dependencies |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data
python data.py

# Train (phases 1-4 automatically)
python train.py \
  --lm_model distilgpt2 \
  --n_samples 2000 \
  --epochs_backbone 3 \
  --epochs_lite 2 \
  --batch_size 16 \
  --output_dir checkpoints

# Evaluate
python evaluate.py \
  --checkpoint checkpoints/univid_system.pt \
  --n_eval_samples 500
```

---

## Fidelity Notes

| Component | Paper | This Reproduction |
|-----------|-------|-------------------|
| Vision Encoder | Large ViT over sampled frames | Random tensor (768-dim placeholder) |
| LM Backbone | Large VLM (InternVL-class) | distilgpt2 (toy) |
| Training Data | Human-refined labels + synthetic | Fully synthetic |
| RAG Index | FAISS ANN | In-memory cosine similarity |
| Scale | Production (ByteDance) | ~2K samples |
| Key Metrics | -42.7% leakage, -37.0% overkill | Relative ensemble improvement |

The reproduction faithfully implements the **pipeline logic and loss formulations** from the paper. Production-scale results require real video data and a large VLM backbone.

---

## Paper Results vs Reproduction

| Metric | Paper (Production) | This Reproduction |
|--------|-------------------|-------------------|
| Violation Leakage Reduction | -42.7% (relative) | ~-5–15% (toy data, expected) |
| Overkill Rate Reduction | -37.0% (relative) | ~-5–15% (toy data, expected) |
| Models replaced | 1,000+ → 1 | 7 policy classifiers → 1 |

The key insight (interpretable captions as intermediate representation + ensemble of risk filter / Lite / RAG) is fully implemented and verifiable on the toy dataset.
