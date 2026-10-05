# KV2

Toy but runnable PyTorch reproduction for **KV²: A Self-Refining KV Cache**.

The original paper's anonymous repository was checked and currently appears empty, so this folder provides a compact implementation of the main method idea:

- proxy token scoring for query-agnostic informative-token selection;
- selective reconstruction over the selected query tokens;
- final eviction scoring and budgeted KV-cache compression;
- iterative self-refinement by repeating the selective-reconstruction pass.

## Files

- `dataset.py` creates synthetic long-context retrieval examples.
- `model.py` implements a tiny attention model plus KV² cache scoring/compression.
- `train.py` trains the toy long-context model.
- `test.py` compares full cache, proxy-only eviction, and KV² eviction.

## Quick start

```bash
python train.py --epochs 5 --output-dir runs/demo
python test.py --checkpoint runs/demo/model.pt --budget-ratio 0.2
```
