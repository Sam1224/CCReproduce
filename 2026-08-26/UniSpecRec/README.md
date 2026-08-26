# UniSpecRec

Toy PyTorch reproduction for **Rethinking Semantic Alignment in LLM-Enhanced Collaborative Filtering: A Spectral Decoupling Approach**.

## What is implemented

This folder keeps the same lightweight pattern as `2026-08-19/CARA`: synthetic data, a compact PyTorch model, training that saves a checkpoint/history, and testing that reloads the checkpoint and prints ranking metrics.

- `data.py`: builds a synthetic implicit-feedback recommendation world with collaborative item factors, noisy LLM-like semantic features, an item graph, and spectrally smoothed semantic features.
- `model.py`: implements a UniSpecRec-style recommender with decoupled collaborative and semantic towers plus late user-conditioned fusion for ranking.
- `train.py`: trains on candidate-set classification and saves `artifacts/unispecrec.pt` plus `artifacts/history.json`.
- `test.py`: reloads the checkpoint and reports HR/NDCG metrics.

## Run

```bash
python3 train.py
python3 test.py
```

## Mapping to the paper

- **Semantic alignment issue** is approximated by raw semantic item features containing cluster signal plus high-frequency noise.
- **Spectral decoupling/smoothing** is approximated by propagating semantic features over a collaborative item graph using normalized adjacency.
- **Decoupled towers** score candidates separately from collaborative IDs and smoothed semantic content.
- **Late fusion** combines the two towers with a user-conditioned gate instead of forcing early semantic-CF alignment.

## Not implemented

- Real LLM embeddings, public recommendation benchmarks, or the exact paper training recipe.
- Large-scale graph eigendecomposition; the toy uses local normalized graph smoothing for a runnable reproduction.
