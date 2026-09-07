# SAM-D2Q (Toy Reproduction)

Lightweight PyTorch reproduction for **SAM-D2Q: Aligning Multimodal Doc2Query with Search Demand and Conversion for E-commerce**.

Paper: https://arxiv.org/abs/2609.04961

## Implemented ideas

- Multimodal product encoder combining product-title tokens and image attribute features.
- Doc2Query sequence decoder for pseudo-query generation.
- Counterfactual visual grounding by masking text attributes and requiring recovery from image features.
- Business-aligned reinforcement step using demand, conversion, and information-gain rewards.
- Toy retrieval evaluation that measures generated-query coverage and commercial utility.

## Quickstart

```bash
pip install -r requirements.txt
python3 train.py --epochs 3
python3 test.py --checkpoint outputs/samd2q.pt
```

## Notes

The production AliExpress logs, CPV/CVR rewards, and deployed indexing system are not public. This reproduction uses synthetic catalog/query examples while preserving the staged SFT + counterfactual augmentation + preference alignment pipeline.
