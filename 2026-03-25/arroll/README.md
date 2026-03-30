# arrol (toy reproduction)

This folder is a **lightweight, runnable reproduction skeleton** of the core idea in:

- **Prune as You Generate: Online Rollout Pruning for Faster and Better RLVR** (arXiv:2603.24840)

## What this reproduction covers

The paper targets RLVR-style training where you generate many rollouts, score them with a **verifiable reward**, and update the policy. A key insight is that you can **prune hopeless rollouts online during generation** (before finishing the full sequence), saving compute and sometimes improving training signal.

This toy implementation demonstrates the mechanism on a simple *verifiable arithmetic* task:

- The agent must generate a fixed-length sequence of digits whose sum equals a target.
- Reward is **1** iff the final sum matches the target (otherwise 0).
- Online rollout pruning drops partial sequences that can no longer reach the target.

We track **tokens sampled** as a proxy for generation compute, and compare training with and without pruning.

## Quickstart

```bash
cd 2026-03-25/arroll
python3 -m pip install -r requirements.txt
python3 train.py
```

## Files

- `toy_task.py`: verifiable reward task + online feasibility pruning
- `policy.py`: small GRU policy over digits conditioned on target
- `rollout.py`: rollout generators (baseline sampling vs pruned sampling)
- `train.py`: REINFORCE training loop + compute accounting

## Notes / limitations

- This is not a full LLM RLVR setup; it’s a minimal reference for **online rollout pruning** logic.
- Reward is perfectly verifiable; in real RLVR, verification comes from executors, unit tests, or structured checkers.
