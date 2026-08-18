# SMEO

Toy PyTorch reproduction for **Sequential Multimodal Evidence Optimization for Product Media Ranking in E-Commerce**.

## What is implemented

This folder reproduces the paper's two-stage intuition with a lightweight synthetic setup.

- `data.py`: generates product sessions with images, videos, and 3D assets; builds prefix-utility and ranking datasets.
- `model.py`: implements a trajectory utility model and a survival-aware ranker.
- `train.py`: trains stage 1 (utility estimation) and stage 2 (autoregressive ranking).
- `test.py`: compares learned ranking against a freshness baseline using swipe efficiency.

## Run

```bash
python3 train.py
python3 test.py
```

Artifacts are written to `artifacts/`.

## Mapping to the paper

- **Trajectory utility modeling** is approximated with prefix utility regression.
- **Sequential evidence optimization** is approximated with an autoregressive next-asset classifier.
- **Survival weighting** is simplified into a swipe-efficiency evaluation target instead of full doubly robust OPE.

## Not implemented

The industrial-scale off-policy estimator, real product sessions, and online serving integration are not publicly available, so this reproduction focuses on the algorithmic structure rather than production fidelity.
