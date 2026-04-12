# Unified Supervision for Sponsored Retrieval (Toy Reproduction)

Paper: **Unified Supervision for Walmart’s Sponsored Search Retrieval via Joint Semantic Relevance and Behavioral Engagement Modeling** (arXiv:2604.07930)

This folder contains a runnable PyTorch *toy reproduction* of the paper’s core idea:
- Engagement-only supervision is noisy/sparse.
- Use **semantic relevance** as primary supervision (teacher-style graded labels).
- Add **multi-channel retrieval priors** (to surface hard negatives / encode production patterns).
- Use **engagement only as a preference signal among semantically relevant items**.

## What is implemented
- Synthetic sponsored-search retrieval dataset generator (queries/items with latent topics).
- Three training targets:
  - `eng_only`: engagement-only supervision.
  - `rel_only`: relevance + retrieval-prior supervision.
  - `rel_eng`: unified supervision (relevance primary + prior + engagement among relevant).
- A simple bi-encoder retriever trained with a **listwise KL distillation** loss between:
  - predicted ranking distribution (softmax over cosine scores)
  - target distribution (softmax over unified target scores)
- Offline metrics (K=25): AvgRelevance@25, Precision@25, NDCG@25.

## What is simplified
- Real system uses cross-encoder cascades, FAISS, large traffic logs, debiasing, and online A/B tests.
- This reproduction uses synthetic data but keeps the *supervision composition logic*.

## Run
```bash
pip install -r requirements.txt
python3 train.py --epochs 5
python3 test.py  --checkpoint checkpoints/rel_eng.pt
```

The `train.py` script prints a side-by-side offline comparison of the three variants.
