# YuvionVL — Code Reproduction

Toy-but-runnable reproduction of **Yuvion VL: A Multimodal Foundation Model for Adversarial Content and AI Safety** (arXiv:2606.25034).

## File Structure

```
YuvionVL/
├── data.py         # SafetyDataset, ConfusionPairDataset, YVREDataset
├── model.py        # YuvionVLModel, C2FTLoss, SafetyGate
├── train.py        # SFT + C2FT training pipeline
├── eval_yvre.py    # YVRE three-level evaluation
└── README.md
```

## Quick Start

```bash
# SFT + C2FT training (toy data)
python train.py --mode both --epochs_sft 5 --epochs_c2ft 3 --embed_dim 256

# YVRE evaluation on all levels
python eval_yvre.py --level all

# Evaluate with checkpoint
python eval_yvre.py --level 3 --checkpoint checkpoints/yuvion_vl_toy.pt
```

## Architecture Summary

```
Input: text + optional image
         ↓
TextEncoder (simplified Transformer)   VisionEncoder (simplified ViT)
         ↓                                        ↓
         └─────────── MultimodalFusion ───────────┘
                              ↓
                    SafetyHead (binary: safe/violation)
                    ViolationType head (5-class)
                    ReasoningProj (CoT feature)
```

## C2FT (Confuse-then-Contrast Fine-Tuning)

```
Phase 1 (SFT):  standard cross-entropy on (text/image, label) pairs
Phase 2 (C2FT): online confusion mining → contrastive push-apart
                Total loss = (1-α) · CE(anchor) + α · InfoNCE(anchor, confused)
```

The paper's full C2FT uses model-specific confusion pairs mined from inference failures and applies multi-image joint contrastive supervision. This reproduction approximates with pre-defined confusion pairs.

## Key Paper Claims (reproduced in spirit)

| Claim | Paper | This Reproduction |
|-------|-------|------------------|
| Safety classification | SOTA on YVRE | Binary safety head |
| Adversarial data flywheel | Continuous mining + augmentation | Toy augmentation of hard examples |
| C2FT confusion mining | Online model-failure mining | Pre-defined confusion pairs |
| E-commerce governance | Logo, brand, category, price | YVREDataset Level 3 examples |
| Model scale | 8B / 32B Dense | Small toy model |

## Notes

- Full reproduction requires a large pretrained MLLM backbone (8B+ parameters) and proprietary e-commerce governance data.
- This reproduction demonstrates the C2FT training paradigm and YVRE evaluation framework at toy scale, with interface-aligned data and model structures.
- For production: swap `TextEncoder`/`VisionEncoder` with a full LLM backbone (e.g., Qwen-VL, InternVL) and use the data flywheel pipeline from the paper.
