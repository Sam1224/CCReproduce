# TAGR

Toy PyTorch reproduction for **TAGR: Temporally Adaptive Generative Recommendation for Industrial Live-Streaming Advertising** (arXiv:2608.24034).

## What is implemented

This folder contains a compact, self-contained approximation of the paper's core ideas in a synthetic live-stream advertising environment.

- `data.py`: creates users, ads, live rooms, time-varying live states, behavior sequences, candidate ads, labels, and online-preference proxy rewards.
- `model.py`: implements a small TAGR-style ranker with:
  - **dynamic live-state tokens** from room stage, focus product category, promotion level, and freshness;
  - an **intent-aware sequence encoder** using GRU history encoding plus live-state-conditioned attention;
  - a generative preference vector that scores ad candidates.
- `train.py`: trains the model, adds a lightweight online-preference proxy term to cross-entropy, and saves checkpoint/history.
- `test.py`: loads the checkpoint and prints HR/NDCG and proxy-reward metrics.

## Run

```bash
python3 train.py
python3 test.py
```

Artifacts are written to `artifacts/` after training.

## Mapping to the paper

- Dynamic live-state tokenization approximates live-room inventory/script/promotion drift.
- Intent-aware sequence encoding approximates user intent after entering a live stream.
- The online-preference proxy loss approximates intermittent online preference optimization with synthetic reward signals.

## Not implemented

- Industrial live-stream logs, production generative retrieval, and true online updates.
- Exact TAGR architecture, serving constraints, and business multi-objective calibration.
