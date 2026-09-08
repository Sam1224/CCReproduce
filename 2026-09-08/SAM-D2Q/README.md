# SAM-D2Q toy reproduction

This folder provides a **toy but runnable** PyTorch reproduction of **SAM-D2Q: Aligning Multimodal Doc2Query with Search Demand and Conversion for E-commerce**.

## What is implemented
- Information-gain filtering that keeps only expansion targets not already exposed in product titles
- Counterfactual visual augmentation by masking title-side visual clues while preserving image attributes
- A multimodal document-to-query model that fuses title tokens with image-side attributes
- Reward-weighted preference alignment toward search relevance, conversion value, and visual grounding
- Synthetic offline search evaluation comparing baseline title indexing vs. augmented indexing

## What is simplified
- Real product images are replaced by structured visual-attribute vectors.
- Pseudo-query generation is formulated as multi-label expansion-token prediction over a fixed vocabulary instead of free-form decoding.
- The RL stage is approximated by a reward-distribution alignment loss rather than online policy optimization.
- The production AliExpress stack is replaced by a synthetic catalog and search benchmark that preserve the paper's supervision shape.

## Files
- `data.py`: synthetic e-commerce products, augmentation logic, and search benchmark
- `model.py`: SAM-D2Q multimodal encoder and reward-alignment objective
- `train.py`: trains the toy model and saves `sam_d2q.pt`
- `test.py`: evaluates search lift after offline document expansion

## Run
```bash
python train.py
python test.py
```

## Expected outcome
The toy model should learn to recover hidden visual attributes and high-value demand terms, then improve recall and weighted search quality when those tokens are injected into the offline index.
