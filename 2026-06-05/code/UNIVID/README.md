# UNIVID (toy reproduction)

- Paper: **UNIVID: Unified Vision-Language Model for Video Moderation** (arXiv:2606.05748)
- Affiliation: ByteDance
- Core idea:
  - Traditional video moderation uses 1000+ fragmented policy-specific classifiers.
  - UNIVID trains a **single VLM backbone** that generates **policy-aware captions** as interpretable intermediate representations.
  - Captions describe which policy rules are potentially violated → violation decision derived from caption.
  - Training mixes expert human-refined labels with LLM-generated synthetic data.

This folder implements a **toy but runnable** pipeline that demonstrates the core UNIVID architecture:

- **Multi-frame video encoding** (simulated with image feature extractor)
- **Policy-conditioned caption generation** (each policy rule as a prefix condition)
- **Violation detection** from caption via a lightweight classifier head
- **Mixed-label training** (human + synthetic labels)

> Notes
>
> - Original UNIVID uses a large-scale VLM (e.g., InternVL/LLaVA family) fine-tuned at ByteDance scale.
> - This toy version uses a small transformer + MLP to demonstrate the interface and training loop.
> - Key interfaces are faithfully reproduced: policy-conditioned generation, caption-based violation head, mixed data training.

## Quickstart

```bash
pip install -r requirements.txt

# Generate synthetic dataset
python data.py --n-videos 500 --out data/univid_toy.pt

# Train UNIVID toy model
python train.py --data data/univid_toy.pt --epochs 10 --out runs/univid.pt

# Evaluate (violation F1, overkill rate, leakage rate)
python evaluate.py --ckpt runs/univid.pt --data data/univid_toy.pt
```

## Files

- `data.py`: Synthetic video dataset with policy-conditioned annotations and noise injection.
- `model.py`: UNIVID toy model (visual encoder + policy-conditioned caption head + violation classifier).
- `train.py`: Mixed-label training with human and synthetic annotation weighting.
- `evaluate.py`: Violation F1, leakage rate, overkill rate evaluation.
- `requirements.txt`: Dependencies.
