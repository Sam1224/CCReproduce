# ARGUS — Code Reproduction

**Paper:** ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement with Adversarial Umpiring  
**arXiv:** https://arxiv.org/abs/2605.02200  
**Score:** 84 / 100

## Overview

This reproduction implements the three-stage ARGUS framework for policy-adaptive online advertising governance.

```
Stage I   — Policy Seeding:               VLM fine-tuned on new policy data via supervised learning
Stage II  — Adversarial Label Rectification: Prosecutor-Defender-Umpire (PDU) + REINFORCE RL
Stage III — Latent Knowledge Discovery:   Hard sample mining + adversarial augmentation fine-tuning
```

## Files

| File | Description |
|------|-------------|
| `data.py` | Toy ad governance dataset + RAG-based policy knowledge base |
| `model.py` | PolicyVLM backbone + PDU architecture (Prosecutor, Defender, Umpire agents) + HardSampleMiner |
| `train.py` | Three-stage training pipeline |
| `evaluate.py` | Evaluation with historical recall (gray-area) as primary metric |
| `requirements.txt` | Dependencies |

## Quick Start

```bash
pip install -r requirements.txt

# Stage I: Policy Seeding
python train.py --stage 1 --policy_version v2_edu_anxiety --epochs 3

# Stage II: Adversarial Label Rectification (RL)
python train.py --stage 2 --policy_version v2_edu_anxiety --epochs 2

# Stage III: Latent Knowledge Discovery
python train.py --stage 3 --policy_version v2_edu_anxiety --epochs 2

# Evaluate
python evaluate.py --policy_version v2_edu_anxiety
python evaluate.py --policy_version v2_edu_anxiety --use_pdu  # PDU-augmented
```

## Policy Versions

| Version | Description |
|---------|-------------|
| `v1_baseline` | Standard false-claim / urgency policies |
| `v2_edu_anxiety` | Educational anxiety restrictions (child performance) |
| `v3_appearance_anxiety` | Appearance anxiety restrictions (body/skin) |

## Architecture Notes

The PDU cycle (Stage II):
1. **Prosecutor** classifies with a low threshold (strict inspector)
2. **Defender** challenges with a high threshold (benign interpretation)
3. **Umpire** adjudicates with RAG-retrieved policy context → produces RL reward

The RL reward signal is proportional to umpire confidence, ensuring gray-area samples produce weaker gradient updates.

## Production Adaptation

To use real VLMs:
1. Replace `PolicyVLM.vision_encoder` with a VLM backbone (e.g., Qwen2-VL, InternVL-2)
2. Replace `PolicyVLM.encode_text` with the VLM's text processing
3. Replace `PolicyKnowledgeBase.retrieve` with FAISS dense retrieval over sentence embeddings
4. Replace REINFORCE with PPO/GRPO for stable RL training
