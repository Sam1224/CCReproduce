# UniScale (Toy Reproduction)

This folder is a *lightweight / toy* reproduction of the core ideas described in:

- **UniScale: Synergistic Entire Space Data and Model Scaling for Search Ranking** (arXiv:2603.24226)
  - https://arxiv.org/abs/2603.24226

The original paper reports an industrial-scale e-commerce search ranking system (Taobao). The full production system, data, and training setup are not publicly available; this reproduction focuses on implementing the **key conceptual mechanisms** with a runnable PyTorch pipeline.

## What’s implemented

### 1) ES³ (Entire-Space Sample System) — toy version
Implemented in `data/es3_sampler.py`.

- **Unexposed candidate expansion**: for each (query, clicked/purchased item) we sample additional *unexposed* candidates from the item pool as negatives.
- **Hierarchical label attribution** (toy): we generate 2 labels `click` and `purchase` with the constraint `purchase => click`. The sampler can optionally upweight higher-level labels.
- **Cross-domain “searchification”** (toy): we create pseudo-search examples by converting interaction events from another domain into (query-like) token sequences.

### 2) HHSFT (Heterogeneous Hierarchical Sample Fusion Transformer) — toy version
Implemented in `model/hhsft.py`.

- Tokenizes a sample into **segments**: `user`, `query`, `item`.
- Adds **domain embeddings** to represent heterogeneous distributions.
- Uses a Transformer encoder with a **domain-gated feed-forward** (a simple 2-expert mixture) to reduce negative transfer.

## Quickstart

From repo root:

```bash
cd 2026-03-30/UniScale
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train & evaluate
python3 train.py --epochs 3 --use_es3 1
```

You should see printed metrics:
- **AUC** (global)
- **GAUC** (group AUC by query-id)

## Notes / gaps vs the original paper

- This code does **not** reproduce the proprietary ES³ data pipelines, the full HHSFT architecture, or Taobao-scale A/B testing.
- The goal is to provide a faithful, runnable skeleton that maps the paper’s core ideas into code:
  - Entire-space sampling to fight exposure bias
  - Heterogeneity-aware model design to unlock gains from scaled, mixed data

## Pseudocode (high level)

**ES³** (toy)

```
for each positive (q, i_pos):
  negatives <- sample_unexposed_items(q, K)
  emit (q, i_pos, y=1)
  emit (q, i_neg, y=0) for i_neg in negatives
```

**HHSFT** (toy)

```
x = embed([user_tokens, query_tokens, item_tokens]) + segment_embed + domain_embed
x = Transformer(x) with domain-gated FFN
logit = MLP(pool(x))
```
