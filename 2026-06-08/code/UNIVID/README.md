# UNIVID — Code Reproduction

Faithful reproduction of the core ideas in:

> **UNIVID: Unified Vision-Language Model for Video Moderation**  
> Kejuan Yang et al., ByteDance, arXiv 2606.05748, June 2026

## Files

```
UNIVID/
├── README.md                  # this file
├── model.py                   # UNIVID VLM architecture
├── dataset.py                 # Toy dataset / data interface
├── train.py                   # Training script (SFT + instruction tune)
├── pipeline.py                # Full 3-stage moderation pipeline
└── evaluate.py                # Evaluation script
```

## Quick Start

```bash
pip install torch transformers datasets peft
python train.py --data_path data/toy_videos --epochs 3
python pipeline.py --video_path sample.mp4 --policy_path policies/sample_policy.txt
```

## Architecture

```
Video Frames + Policy Text
         │
   ┌─────▼──────────────────────────────────┐
   │  UNIVID VLM                            │
   │  ┌──────────┐  ┌────────────────────┐  │
   │  │ Visual   │  │  LLM Backbone      │  │
   │  │ Encoder  │─▶│  (Qwen/LLaMA)      │  │
   │  └──────────┘  └────────────────────┘  │
   │       policy tokens ──────────────▶    │
   └─────────────────────────────────────────┘
         │ policy-aware caption
         │ caption embedding
         ▼
   ┌─────────────────────────────────────────┐
   │  3-Stage Moderation Pipeline            │
   │  Stage 1: Risk Filter (embedding sim)   │
   │  Stage 2: Moderation Actor (cls head)   │
   │  Stage 3: Trend Governance (clustering) │
   └─────────────────────────────────────────┘
```
