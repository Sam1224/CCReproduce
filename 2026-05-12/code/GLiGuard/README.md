# GLiGuard — Code Reproduction

Reproduction of **"GLiGuard: Schema-Conditioned Classification for LLM Safeguard"**  
arXiv: https://arxiv.org/abs/2605.07982

## Architecture Overview

```
Input Sequence:
[SCHEMA: task_name] [LABEL: label_1 | description_1] ... [LABEL: label_N | description_N]
[SEP] <Content to classify (prompt or response)> [SEP]

↓  Frozen / Fine-tuned DeBERTa-v3 / ModernBERT bidirectional encoder  ↓

[CLS] representation  →  Per-label binary classifiers  →  Safety decisions
       ↑
  Shared hidden states from FULL bidirectional attention over schema + content
```

The key insight from GLiNER2: encode the **label semantics into the input** rather than keeping them external. This lets the encoder attend over label descriptions and content jointly, enabling soft label-context co-attention.

## Files

| File | Purpose |
|------|---------|
| `data.py` | Toy dataset builder; maps public safety benchmarks schema to GLiGuard format |
| `model.py` | GLiGuard model — schema-conditioned bidirectional encoder + per-label classifiers |
| `train.py` | Training loop with multi-label cross-entropy |
| `eval.py` | Evaluation script — F1 per dimension, macro F1, throughput benchmark |
| `schemas.py` | Pre-built safety schemas (harm categories + jailbreak types) |
| `requirements.txt` | Minimal dependencies |

## Quick Start

```bash
pip install -r requirements.txt

# Generate toy training data
python data.py --output toy_data.jsonl

# Train
python train.py --data toy_data.jsonl --epochs 3 --output ./checkpoints

# Evaluate
python eval.py --checkpoint ./checkpoints/best.pt --data toy_data.jsonl
```

## Differences from Paper

- Uses `DeBERTa-v3-base` (0.18B) or `ModernBERT-base` as backbone (paper uses GLiNER2's backbone, an adapted DeBERTa-v3)
- Toy dataset uses synthetic safety examples matching the paper's schema format
- Multi-label classification head per schema slot; paper trains with held-out safety benchmark data
- Throughput benchmark script shows latency vs. 7B autoregressive baseline (simulated)
