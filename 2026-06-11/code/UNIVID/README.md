# UNIVID: Unified Vision-Language Model for Video Moderation

Reproduction of: [UNIVID: Unified Vision-Language Model for Video Moderation](https://arxiv.org/abs/2606.05748)  
ByteDance, June 4, 2026

## Structure

```
UNIVID/
├── model.py          # VLM backbone + policy-aware caption head
├── pipeline.py       # Three-stage moderation pipeline
├── data.py           # Dataset loading (toy data interface-aligned)
├── train.py          # Fine-tuning script
├── eval.py           # Evaluation script
└── README.md
```

## Setup

```bash
pip install torch transformers accelerate pillow einops
```

## Quick Start

```bash
# Train UNIVID caption model
python train.py --stage caption --data data/toy_videos/ --epochs 3

# Run full moderation pipeline
python pipeline.py --video_path sample.mp4 --policy policy.json

# Evaluate on held-out set
python eval.py --data data/eval/ --model checkpoints/univid_best.pt
```
