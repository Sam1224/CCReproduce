# SARM — Code Reproduction

Paper: "SARM: LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking"  
arXiv: 2602.09401 | Kuaishou Technology & Chinese Academy of Sciences

## Key Idea

Semantic anchors are **learnable text token embeddings** that represent live-stream content semantics, jointly optimized with the ranking objective. LLM-generated descriptions initialize the anchors offline; online ranking uses cached anchor representations.

## Structure

```
SARM/
├── model.py   # SemanticAnchor, LiveStreamContentEncoder, SARMRankingModel, SARMTrainingLoss
├── train.py   # Two-stage training: offline LLM anchor generation + end-to-end ranking
└── README.md
```

## Setup

```bash
pip install torch transformers
```

## Quick Start

```bash
python train.py --epochs 10 --anchor_tokens 8 --batch_size 64
```

## Architecture Highlights

| Component | Description |
|-----------|-------------|
| `SemanticAnchor` | K learnable tokens + self-attention pooling; initialized from LLM embeddings |
| `LiveStreamContentEncoder` | Fuses text (BERT), visual (ResNet/CLIP), interaction signals |
| `SARMRankingModel` | Combines content, anchor, user, and context representations |
| `SARMTrainingLoss` | BPR pairwise + BCE point-wise joint loss |

## Anchor Granularity (from paper ablation)

| Granularity | Performance | Recommended |
|-------------|-------------|-------------|
| Session-level | Lowest | No |
| **Topic-level** | **Highest** | **Yes** |
| Product-level | Middle | No |

## Separate Learning Rates

Semantic anchors benefit from a higher learning rate than the rest of the model:
- `anchor_lr`: 5e-4 (default)
- `model_lr`: 1e-4 (default)
