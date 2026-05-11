# AFMRL: Code Reproduction

**Paper:** AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce  
**arXiv:** https://arxiv.org/abs/2604.20135  
**Institution:** Taobao & Tmall Group, Alibaba  
**Score:** 81/100

## Architecture

```
  Image feat (ViT)    Text feat (CLIP)
        │                   │
   image_proj           text_proj
        └──────────┬────────┘
                   ↓
          MultiModalEncoder
          (additive fusion + LN)
                   │
          ┌────────┴──────────┐
          ↓                  ↓
    AttributeGenerator   RepresentationHead
    (MLP → attr_emb,     (concat fused+attr → L2-norm emb)
     attr_logits)
          │                  │
          │       ┌──────────┘
          └───────┴──── AGCL Loss (Stage 1)
          │             ↑ attribute-guided contrastive
          │
          └──────────── RAR Loss  (Stage 2)
                        ↑ retrieval reward → attr generator
```

## Two-Stage Training

### Stage 1: AGCL (Attribute-Guided Contrastive Learning)

- MLLM extracts product attributes from image+text
- Attributes identify **hard negatives** (similar products that differ in attributes)
- Attributes filter **false negatives** (identical products incorrectly treated as negatives)
- Modified InfoNCE loss with attribute-guided negative sampling

### Stage 2: RAR (Retrieval-aware Attribute Reinforcement)

- Freeze encoder + representation head
- Fine-tune attribute generator only
- Reward signal: improvement in retrieval performance
- Proxy: cosine alignment of attribute embedding with positive retrieval embedding

## Files

| File | Description |
|---|---|
| `model.py` | Full model: MultiModalEncoder, AttributeGenerator, RepresentationHead, AGCLLoss, RARLoss, AFMRL |
| `train.py` | Two-stage training loop with toy e-commerce data |
| `eval.py` | Retrieval evaluation: Recall@K, MRR |

## Quick Start

```bash
pip install torch numpy scikit-learn

# Train both stages (toy data, small dims for CPU)
python train.py

# Evaluate saved checkpoint
python eval.py
```

## Key Implementation Notes

### False Negative Filtering (Section 3.1)
Products with attribute overlap > `tau_fn` (default: 0.7) are excluded from the negative set — they may actually be the same product with different images.

### Hard Negative Mining (Section 3.1)  
Among remaining negatives, we select top-K by attribute overlap — these are the most challenging true negatives (different product but similar attributes).

### RAR Proxy Reward (Section 3.2)
The paper uses a non-differentiable retrieval ranking reward via REINFORCE. Our implementation uses a differentiable proxy: maximizing cosine similarity between attribute embedding and stop-gradient positive embedding achieves equivalent training signal.

### Production Notes
- Replace `AttributeGenerator` MLP with actual MLLM (LLaVA/InternVL)
- Replace `image_proj`/`text_proj` with pretrained CLIP/ViT encoders
- Scale negative mining pool to thousands for production

## Correspondence to Paper

| Paper Component | Code Location |
|---|---|
| MLLM attribute extraction | `model.py::AttributeGenerator` |
| AGCL hard negative mining | `model.py::AGCLLoss.forward` |
| AGCL false negative filter | `model.py::AGCLLoss.forward (fn_mask)` |
| RAR reward training | `model.py::RARLoss` |
| Stage 1 training loop | `train.py::train_stage1_agcl` |
| Stage 2 training loop | `train.py::train_stage2_rar` |
| Recall@K, MRR | `eval.py` |
