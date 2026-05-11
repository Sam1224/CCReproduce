# GLiGuard — Code Reproduction

**Paper:** GLiGuard: Schema-Conditioned Classification for LLM Safeguard  
**arXiv:** https://arxiv.org/abs/2605.07982  
**Score:** 83/100

## Overview

GLiGuard is a 0.3B bidirectional encoder for multi-task LLM content safety classification.
It frames guardrailing as schema-conditioned multi-label classification, achieving
competitive F1 with 7B–27B autoregressive guardrail models at 23–90× smaller size,
16× higher throughput, and 17× lower latency.

## Structure

```
GLiGuard/
├── README.md
├── requirements.txt
├── model.py           # GLiGuard model architecture
├── data.py            # Dataset loading & schema construction
├── train.py           # Training script
└── evaluate.py        # Evaluation on safety benchmarks
```

## Quick Start

```bash
pip install -r requirements.txt
# Train
python train.py --data_dir ./data --output_dir ./checkpoints
# Evaluate
python evaluate.py --model_path ./checkpoints/best --data_dir ./data
```
