# GradCIR reproduction

Implements a compact PyTorch reproduction of the paper's core pipeline: composed image-query encoding, product encoding, four-level graded relevance supervision, hierarchy-aware angular loss, hard-negative-ready batch scoring, and NDCG@K evaluation on a toy dataset with the same interfaces as the training script.

Run:

```bash
python train.py --epochs 3
python test.py
```
