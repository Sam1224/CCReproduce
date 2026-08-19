# DataDPO

Toy PyTorch reproduction for **Data-DPO: Direct Preference Optimization for Target Model Data Selection in LLM Post-Training** (arXiv:2608.16926).

## What is implemented

This folder reproduces the paper's *target-aware data selection* idea with a small synthetic classification setting.

- `data.py`: generates a candidate SFT pool containing in-domain and out-of-domain samples plus a clean validation/test set.
- `model.py`: defines a lightweight target model (linear classifier) and a reward model (MLP) for learning data preferences.
- `train.py`:
  - performs one-step probing to estimate per-sample "utility" for the current target model;
  - converts utilities into pairwise preferences and trains a reward model with a DPO-style pairwise objective;
  - selects a subset with reward + external quality + marginal diversity;
  - fine-tunes the target model on the selected subset and compares against random subset and full-data training.
- `test.py`: prints the saved results.

## Run

```bash
python3 train.py
python3 test.py
```

Artifacts are written to `artifacts/`.

## Mapping to the paper

- **One-step probing** is implemented as a single SGD step on one sample followed by validation-loss improvement.
- **Data preference** is derived from pairwise utility comparisons.
- **DPO reward learning** is implemented as a pairwise logistic objective on reward differences.
- **Selection** combines reward score, a toy external quality score, and a greedy diversity term.

## Not implemented

- Vision-Flan / LLaVA-CoT pipelines and multimodal training.
- Large-scale distributed probing and production caching.
