# NT-SSM (toy reproduction)

- Paper: **Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy** (ICML 2026, arXiv:2605.24015)
- Core idea: unfold LightGCN’s score into *neighbor-pair aggregation* and design a contrastive objective (NT-SSM) that (1) accounts for both user- and item-side structural similarity and (2) uses **neighbor-type-specific coefficients** to modulate update dynamics across UU / II / UI / IU neighbor-pair types.

This folder implements a **toy but runnable** reproduction:

- A small bipartite user-item interaction graph generator.
- LightGCN propagation weights `\tilde{S}` (Eq. in Sec 3) computed exactly for the toy graph.
- **SSM** baseline.
- **NT-SSM** loss (Sec 6) using type-decomposed negative similarities.
- Offline evaluation: Recall@K / NDCG@K.

> Note: The paper’s referenced code link is currently inaccessible (GitHub 404 as of 2026-05-26), so this is a from-scratch implementation based directly on the paper’s equations.

## Quickstart

```bash
python -m pip install -r requirements.txt

# Train SSM baseline
python train.py --loss ssm --epochs 50

# Train NT-SSM
python train.py --loss nt_ssm --epochs 50

# Evaluate a checkpoint
python test.py --ckpt runs/nt_ssm.pt
```

## Files

- `data.py`: toy CF dataset builder (train/val/test split).
- `graph.py`: build adjacency and compute `\tilde{S}`.
- `model.py`: LightGCN with fixed `\tilde{S}`.
- `losses.py`: SSM + NT-SSM.
- `train.py`: training loop.
- `test.py`: evaluation.
