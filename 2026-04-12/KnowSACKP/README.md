# KnowSACKP (Toy Reproduction)

Paper: **Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders** (arXiv:2604.07825)

This folder provides a runnable toy reproduction of the paper’s core mechanism:
- Item-level **knowledge gap** exists (long-tail items are less known).
- **Uniform augmentation** wastes context and can even hurt.
- Use **Comparative Knowledge Probing (CKP)** to estimate which items are under-known, and only augment those.

## What is implemented (toy)
- A synthetic recommendation dataset with user histories, candidate sets, and a ground-truth next-item.
- A `MockLLM` that has deliberate *knowledge gaps* (popular items have better parametric representations, tail items are noisy).
- A CKP-like scorer that estimates an item’s knowledge quality via pairwise comparisons against anchors.
- Augmentation strategies:
  - `no_augment`: use parametric (noisy) item representations.
  - `uniform`: always use external attribute/text representation.
  - `selective`: use CKP to augment only low-knowledge items.

## What is simplified
- The real method uses a real LLM ranker and real external knowledge sources.
- Here we simulate those with a small neural representation module and a synthetic external attribute embedding.

## Run
```bash
pip install -r requirements.txt
python3 train.py --items 500 --users 2000 --candidates 20
python3 test.py  --cache caches/ckp_cache.pt
```

`train.py` builds CKP scores and writes a cache; `test.py` evaluates Recall@1 under different augmentation policies.
