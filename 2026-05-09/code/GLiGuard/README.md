# GLiGuard — Code Reproduction

**Paper:** GLiGuard: Schema-Conditioned Classification for LLM Safeguard  
**arXiv:** https://arxiv.org/abs/2605.07982  
**Score:** 80 / 100

## Overview

GLiGuard is a 0.3B-parameter schema-conditioned bidirectional encoder for multi-aspect LLM content moderation. It evaluates all safety dimensions in a single forward pass, achieving 16× throughput and 17× lower latency vs. 7B–27B autoregressive guardrails, while maintaining competitive F1.

## Key Architecture

```
Input: [PROMPT] + [RESPONSE]
         ↓
Bidirectional Encoder (GLiNER2 backbone, 0.3B)
         ↓
Schema-conditioned Cross-Attention
 (schema dim embeddings attend to encoder hidden states)
         ↓
Parallel Classification Head → 25 binary outputs in ONE forward pass
  ├── prompt_safety
  ├── response_safety
  ├── refusal_detection
  ├── violence, hate_speech, sexual_content ... (14 harm categories)
  └── role_play, hypothetical, ... (11 jailbreak strategies)
```

## Files

| File | Description |
|------|-------------|
| `data.py` | Toy LLM safety dataset with multi-aspect schema and label generation |
| `model.py` | GLiGuard model: SchemaEmbedder + BidirectionalEncoder + classification head |
| `train.py` | Multi-task training with parallel per-dim BCE losses |
| `evaluate.py` | Per-dimension F1/precision/recall + throughput/latency benchmarking |
| `requirements.txt` | Dependencies |

## Quick Start

```bash
pip install -r requirements.txt

# Train
python train.py --epochs 5 --lr 2e-4 --batch_size 16

# Evaluate on specific schema dimensions
python evaluate.py --checkpoint checkpoints/gliguard_best.pt \
    --schema prompt_safety response_safety violence hate_speech role_play
```

## Schema Dimensions

**Meta Tasks (3):** `prompt_safety`, `response_safety`, `refusal_detection`

**Harm Categories (14):** `violence`, `hate_speech`, `sexual_content`, `self_harm`, `illegal_activities`, `misinformation`, `privacy_violation`, `child_safety`, `cybersecurity`, `financial_fraud`, `weapons`, `drug_abuse`, `radicalization`, `harassment`

**Jailbreak Strategies (11):** `role_play`, `hypothetical`, `code_injection`, `prompt_injection`, `token_manipulation`, `context_switching`, `authority_impersonation`, `false_premises`, `multi_step`, `encoded_content`, `adversarial_suffix`

## Production Adaptation

To use the real GLiNER2 backbone:
1. Install: `pip install gliner`
2. Replace `BidirectionalEncoder` with `GLiNER.from_pretrained("urchade/gliner_mediumv2.1")`
3. The schema conditioning module (`SchemaEmbedder` + `schema_cross_attn`) remains the same
4. Fine-tune on safety benchmarks: WildGuard, ToxicChat, XSTest, BeaverTails
