# HybridMod — Code Reproduction

**Paper:** Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching  
**arXiv:** https://arxiv.org/abs/2512.03553  
**Venue:** KDD 2026

## Architecture

```
Input Video Segment (multimodal: text transcript + audio + video frames)
           │
    ┌──────┴──────┐
    │             │
 Pipeline A    Pipeline B
Supervised     Similarity
Classifier     Matcher
    │             │
MLLM Teacher   MLLM Feature
  (offline      Extractor
  distillation) (online)
    │             │
Lightweight    Reference
Classifier      Gallery
    │             │
  Score A      Score B
    │             │
    └──────┬──────┘
    Decision Fusion Layer
           │
     Moderation Decision
```

## Files

- `model.py` — HybridMod model (classifier + similarity pipeline)
- `mllm_teacher.py` — MLLM teacher for distillation
- `dataset.py` — Toy multimodal dataset
- `train.py` — Training script
- `evaluate.py` — Evaluation script

## Quick Start

```bash
pip install torch transformers pillow numpy
python train.py --epochs 10 --batch_size 8
python evaluate.py --checkpoint checkpoints/best.pt
```
