# ORCA (Toy Reproduction)

This folder contains a **toy, self-contained** PyTorch reproduction of the *core mechanism* of:

**Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning** (arXiv:2604.01170)

> Note: The original paper targets large LLM reasoning trajectories. Here we reproduce the algorithmic ideas in a **synthetic step-wise reasoning** environment (toy dataset), keeping interfaces aligned to a typical training/eval pipeline.

## What is reproduced

- **Confidence probe**: a small MLP that predicts the probability that stopping at step *t* yields a correct answer.
- **Online / test-time training (TTT)**: per-instance gradient step(s) that adapt the probe using noisy pseudo-labels (standing in for the paper’s supervised labels or self-consistency labels).
- **Risk-controlled early stopping**: we calibrate a stopping threshold `tau` on a calibration split such that the resulting *procedure* (including TTT) achieves a target risk `delta`.

## Setup

```bash
pip install -r requirements.txt
```

## Quickstart

Train a base probe, then evaluate **static calibration** vs **ORCA-style TTT calibration**:

```bash
python -m orca.train_probe --out_dir outputs/probe
python -m orca.eval --ckpt outputs/probe/probe.pt --delta 0.10
```

The evaluation prints:
- **Risk**: fraction of incorrect outputs after early stopping.
- **Savings**: fraction of reasoning steps saved compared to always using the full trajectory.

## Notes on mapping to the paper

- The paper’s *reasoning steps* correspond here to a synthetic trajectory of feature vectors `x[t]` with an underlying correctness probability that typically increases with `t`.
- The paper’s **TTT calibration module** is represented here by a copy of the probe updated per-instance on a short prefix using pseudo-labels `c[t]`.
- Conformal/LTT ideas are implemented as **procedure-level threshold selection** on a held-out calibration set.

## Folder structure

- `orca/data.py`: toy dataset generation + dataloaders
- `orca/models.py`: confidence probe
- `orca/train_probe.py`: train base probe
- `orca/eval.py`: calibrate thresholds and evaluate static vs ORCA procedure
- `orca/calibration.py`: threshold calibration helpers

