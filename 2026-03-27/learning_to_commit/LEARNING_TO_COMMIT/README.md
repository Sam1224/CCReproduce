# Learning to Commit (toy reproduction)

This folder is a **runnable reproduction skeleton** for:

- **Learning to Commit: Generating Organic Pull Requests via Online Repository Memory** (arXiv:2603.26664)

The paper’s core claim is that PR acceptance depends on aligning with *repo-specific conventions* (tests, scope, style, and maintainer preferences), not just producing correct code in isolation.

## What this reproduction covers

A minimal closed-loop simulation of “online repo memory → PR generation → acceptance”:

- **Online repository memory**: encode repo conventions into a small vector (e.g., tests required, prefer small diffs, docs required).
- **PR candidates**: generate multiple candidate “patch plans” with features (has tests, diff size, docs updated).
- **Learning to commit**: a PyTorch ranker learns to pick the candidate that would be accepted under the current repo memory.
- **Evaluation**: top-1 accuracy vs. an acceptance oracle; acceptance rate of the chosen PR.

## What is NOT covered (paper-level TODOs)

- Real code diffs across large repos
- LLM-based patch synthesis and iterative refinement
- True online learning across time and maintainer feedback

Those parts are left as TODO notes (extension points in `repo_memory.py` and `patches.py`).

## Quickstart

```bash
cd ccreproduce_repo/2026-03-27/learning_to_commit/LEARNING_TO_COMMIT
python3 -m pip install -r requirements.txt

python3 build_toy_dataset.py --out /tmp/ltc_train.json --n 2000
python3 build_toy_dataset.py --out /tmp/ltc_test.json --n 500 --seed 2

python3 train.py --train /tmp/ltc_train.json --out /tmp/ltc.pt
python3 test.py --test /tmp/ltc_test.json --ckpt /tmp/ltc.pt
```

## Files

- `repo_memory.py`: toy “online repo memory” encoder
- `patches.py`: candidate PR plans + acceptance oracle
- `dataset.py`: dataset schema / loader
- `model.py`: ranker model
- `build_toy_dataset.py`: synthetic data generator
- `train.py`: train ranker
- `test.py`: evaluate / print sample PR plan
