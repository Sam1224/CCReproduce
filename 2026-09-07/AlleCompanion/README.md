# AlleCompanion (Toy Reproduction)

Lightweight PyTorch reproduction for **Beyond Co-purchase Relation: Evolution of Complementary Recommendations at Allegro**.

Paper: https://arxiv.org/abs/2609.05063

## Implemented ideas

- ComCat-style complementary category mapping from expert rules, LLM scores, and behavioral statistics.
- Category-constrained two-tower retrieval model for item-to-item complementarity.
- Noise-aware training over co-purchase pairs with category masks and in-batch negatives.
- Toy candidate retrieval and Hit@K evaluation.

## Quickstart

```bash
pip install -r requirements.txt
python3 train.py --epochs 3
python3 test.py --checkpoint outputs/allecompanion.pt
```

## Notes

Allegro's production clickstream, human-in-loop labels, and online serving stack are not public. This implementation reproduces the paper's controllable retrieval structure with synthetic category/item features.
