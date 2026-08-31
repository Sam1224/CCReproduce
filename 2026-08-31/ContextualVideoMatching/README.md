# Contextual Video Matching toy reproduction

PyTorch reproduction of the core architecture from `Every Article Deserves a Video: Contextual Video Matching for Digital Publishers`.

Implemented pieces:
- LLM-style article-to-hypothetical-video metadata synthesis interface.
- Shared embedding model for articles and videos.
- Language filtering, cosine retrieval, freshness/performance score mixing, and publisher-ready ranking.
- Contrastive training objective for article-video pairs.
- Toy ecommerce/creator-governance dataset, training script, and retrieval smoke test.

Production parts that require external services are represented by explicit interfaces: LLM extraction, vector database indexing, cache, and safety/ad-fit filters.

Run:

```bash
python test.py
python train.py
```
