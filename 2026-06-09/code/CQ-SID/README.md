# CQ-SID + EG-GRPO — Toy Reproduction

Reproduction of the core mechanisms from:

**Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL**
arXiv: 2605.14434

## Key Concepts Implemented

1. **CQ-SID** (`cq_sid.py`): Category-and-Query Constrained Semantic IDs via Residual VQ
2. **EG-GRPO** (`eg_grpo.py`): Expert-Guided Group Relative Policy Optimization
3. **GenerativeRetriever** (`eg_grpo.py`): Policy that generates product codes given a query
4. **Two-stage Training** (`train.py`): Stage 1 (encoder training) + Stage 2 (GRPO retriever)

## Files

| File | Description |
|------|-------------|
| `data.py` | Toy e-commerce data (products with categories, queries, relevance) |
| `cq_sid.py` | CQ-SID encoder: category-aware + query-item contrastive + RQ-VAE |
| `eg_grpo.py` | EG-GRPO: expert-guided RL for generative retrieval |
| `train.py` | Two-stage training script |
| `evaluate.py` | Recall@K evaluation with hierarchical code matching |
| `requirements.txt` | Dependencies |

## Usage

```bash
pip install -r requirements.txt
python train.py       # stage 1 (CQ-SID) + stage 2 (EG-GRPO)
python evaluate.py    # Recall@K evaluation
```

## Paper Formulas Implemented

**CQ-SID Contrastive Losses**:
```
z_p = Encoder(product_text, category_embedding)
z_q = QueryEncoder(query_text)
L_cat = CategoryContrastive(z_p)  # pull same-category products
L_qi  = InfoNCE(z_q, z_p)         # pull query-matching products
L_rq  = RQVAELoss(z_p)            # quantization reconstruction loss
L_total = L_rq + 0.5*L_qi + 0.3*L_cat
```

**EG-GRPO**:
```
For each query, sample n_samples code sequences {c^1, ..., c^n}:
    reward_i = ExpertRanker(query, c^i)
    A_i = (reward_i - mean(rewards)) / std(rewards)  # group-relative advantage
    L_GRPO = -E[A_i * sum_k log pi_theta(c_k | query, c_{<k})]
```

## Limitations vs. Paper

- Toy data with random features (no actual product/query text encoders)
- Simplified expert ranker (random network; paper uses production ranker)
- Small scale (1K products vs. millions in production)
- Category constraints via simple embedding (paper uses hierarchical taxonomy)
