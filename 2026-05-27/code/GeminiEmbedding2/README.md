# Gemini Embedding 2 — Code Reproduction

**Paper:** [Gemini Embedding 2: A Native Multimodal Embedding Model from Gemini](https://arxiv.org/abs/2605.27295)  
**arXiv:** 2605.27295 | **Score:** 84/100

---

## Structure

```
GeminiEmbedding2/
├── model.py       — backbone, tokenizers, multi-task InfoNCE loss
├── dataset.py     — toy datasets for all four modalities
├── train.py       — multi-stage training script
├── evaluate.py    — retrieval, cross-modal, clustering evaluation
└── README.md
```

---

## Quick Start

```bash
# Install dependencies (minimal)
pip install torch torchvision

# Train (toy-scale, ~2 min on CPU)
python train.py --stage 1 --epochs 3 --batch_size 32 --hidden_dim 256 --num_layers 4

# Evaluate
python evaluate.py --checkpoint checkpoints/stage1_final.pt --task all
```

For full-scale training (Gemini-equivalent):
```bash
python train.py --stage 1 --epochs 10 --batch_size 2048 \
    --hidden_dim 4096 --num_layers 32 --lr 2e-4
python train.py --stage 2 --epochs 5 --batch_size 1024 \
    --hidden_dim 4096 --num_layers 32 --lr 5e-5 \
    --checkpoint checkpoints/stage1_final.pt
```

---

## Architecture

```
Input (any modality)
    │
    ▼
┌─────────────────────────────┐
│  Modality Tokenizer         │  Text: BPE embed
│  (Text / Image / Audio / Video)│  Image: ViT patches → linear proj
│                             │  Audio: mel → 1D conv
│                             │  Video: per-frame ViT + temporal pos
└─────────────────────────────┘
    │ + modality-type embedding
    ▼
┌─────────────────────────────┐
│  Shared Transformer         │  12-32 layers (toy: 4)
│  Backbone (Gemini-style)    │  RoPE, SwiGLU, RMSNorm
└─────────────────────────────┘
    │  collect last N layer hidden states
    ▼
┌─────────────────────────────┐
│  Pooling + Projection Head  │  avg(last N layers) → mean pool
│                             │  → 2-layer MLP → L2 normalize
└─────────────────────────────┘
    │
    ▼
  Embedding (D,)  ← e.g. D=768 or 3072
```

---

## Training Stages

| Stage | Data | LR | Batch | Description |
|-------|------|----|-------|-------------|
| 1 | Web-scale text/image pairs | 2e-4 | 2048+ | Broad contrastive pre-training |
| 2 | Domain-specific (retrieval, clustering, QA, etc.) | 5e-5 | 1024 | Multi-task fine-tuning |

---

## Key Equations

**Multi-task InfoNCE loss** (Eq. 3 in paper):

For task t with temperature τ_t:
```
L_t = -1/B × Σ_i log [ exp(q_i · k_i / τ_t) / Σ_j exp(q_i · k_j / τ_t) ]
```

**Mean pooling** over last N layers:
```
h = mean(h_{L-N}, ..., h_L)     # average last N hidden states
e = MeanPool({h_t : t ∈ non-pad})
z = Proj(e)
z_hat = z / ||z||_2             # L2 normalize
```

---

## Limitations of Reproduction

- **Scale**: Toy config uses hidden_dim=256, 4 layers vs. Gemini's multi-billion parameters.
- **Video/Audio tokenizers**: Simplified; real model uses learnable temporal attention for video and longer mel windows for audio.
- **Training data**: Random toy data; real training uses web-scale image-text, audio-text, video-text pairs.
- **Stage 2 tasks**: Toy text-text only; real Stage 2 covers retrieval (NQ, TriviaQA), clustering (MTEB), STS, visual QA, etc.
- **Distributed training**: Not implemented; real training uses large-scale TPU pod training.
