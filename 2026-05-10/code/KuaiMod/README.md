# KuaiMod — Code Reproduction

Reproduction of **"VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform"**  
arXiv: https://arxiv.org/abs/2504.14904  
Authors: Xingyu Lu, Tianke Zhang et al. (Kuaishou Technology)

## Overview

KuaiMod frames content moderation as "common-law case-based reasoning" using a Vision-Language Model (VLM). The VLM is trained to produce Chain-of-Thought reasoning before issuing a moderation verdict, then updated online with live platform feedback.

## Architecture

```
Input Video/Image + Text
        │
   ┌────▼────┐
   │  VLM    │  (Qwen-VL / InternVL / LLaVA backbone)
   │ Encoder │
   └────┬────┘
        │  Visual + Text tokens
   ┌────▼────────────────┐
   │  CoT Reasoning Head │  → Chain-of-Thought rationale
   └────┬────────────────┘
        │
   ┌────▼──────────────┐
   │  Policy Classifier │  → verdict: {safe, violating, borderline}
   └────┬──────────────┘
        │
   ┌────▼──────────────────┐
   │  Online Feedback Loop  │  user reports + reviewer labels → retrain
   └───────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `data.py` | Toy SVP dataset with CoT annotation format |
| `model.py` | KuaiMod VLM wrapper with CoT + classification head |
| `train.py` | Two-stage training: classification warmup → CoT fine-tune |
| `evaluate.py` | Evaluation on SVP-style benchmark |
| `online_feedback.py` | Simulated online feedback refinement loop |
| `requirements.txt` | Python dependencies |

## Quick Start

```bash
pip install -r requirements.txt

# Stage 1: Classification warmup
python train.py --stage warmup --epochs 3

# Stage 2: CoT fine-tuning
python train.py --stage cot --epochs 5

# Evaluate
python evaluate.py --checkpoint checkpoints/best_cot.pt
```

## Notes on Reproduction

- The original model uses a proprietary Kuaishou VLM backbone trained on internal data. We use a publicly available VLM (e.g., InternVL2-2B or LLaVA-1.5-7B) as backbone.
- The SVP Benchmark is internal to Kuaishou. We provide a toy dataset with the same annotation schema.
- The online feedback loop is simulated via a static validation set.
