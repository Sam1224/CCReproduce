# CARA

Toy PyTorch reproduction for **CARA: Cognitive Adaptive Recommendation Agent** (arXiv:2608.16919).

## What is implemented

This folder reproduces the paper's core intuition with a lightweight synthetic recommendation environment.

- `data.py`: builds a synthetic Amazon-like implicit-feedback dataset with user preference constraints and item metadata; provides train/val/test splits.
- `model.py`: implements a 2-stage *candidate filtering → dual-perspective decision* recommender.
  - Candidate filtering approximates coarse preference constraints.
  - Dual decision routes produce *affective* and *rational* scores and fuse them with a user-conditioned gate.
- `train.py`: trains with a boundary-aware sample weighting strategy (a toy proxy of boundary-aware KTO).
- `test.py`: reports HR@1/5 and NDCG@5/10.

## Run

```bash
python3 train.py
python3 test.py
```

Artifacts are written to `artifacts/`.

## Mapping to the paper

- **Candidate filtering** is modeled as coarse top-k gating before full scoring.
- **Affective vs rational decision** is modeled as two scoring heads fused by a gate.
- **Boundary-aware KTO** is approximated by emphasizing *borderline* samples where the model is neither confident nor consistently correct.

## Not implemented

- Real Amazon Reviews text fields and LLM-based instruction tuning.
- The exact KTO formulation and preference data construction.
- Industrial retrieval stacks and multi-objective serving.
