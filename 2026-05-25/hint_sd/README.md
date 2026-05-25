# HINT-SD — Toy Reproduction (PyTorch)

This folder is a **toy but runnable scaffold** inspired by:

**"HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents"** (arXiv:2605.17873)

The paper proposes **targeted hindsight self-distillation**: instead of generating / using feedback at every turn, it first identifies *failure-relevant* action spans from the full trajectory, then distills a feedback-conditioned teacher only on those targeted spans.

This toy reproduction keeps the core training logic:

- sparse terminal success signal (long-horizon)
- hindsight analysis over the full trajectory to pick **where to distill**
- privileged teacher conditioned on hindsight feedback
- student trained **without feedback**, distilling teacher behavior only on selected spans

## Files

- `env.py`: mini grid-world with sparse terminal reward
- `hindsight.py`: toy hindsight analyzer (oracle selects earliest suboptimal step)
- `model.py`: student policy net + feedback-conditioned teacher net
- `train.py`: trains either `hint_sd` (targeted) or `dense` (per-turn feedback baseline)
- `test.py`: evaluation script
- `requirements.txt`

## Quickstart

```bash
cd CCReproduce/2026-05-25/hint_sd
python3 -m pip install -r requirements.txt

# Targeted hindsight distillation (HINT-SD)
python3 train.py --mode hint_sd --epochs 60
python3 test.py --ckpt checkpoints/hint_sd_student.pt

# Dense per-turn baseline (feedback every turn)
python3 train.py --mode dense --epochs 60 --save checkpoints/dense_student.pt
python3 test.py --ckpt checkpoints/dense_student.pt
```

## Notes on faithfulness

- **Kept**: “select where to distill” + feedback-conditioned teacher + distill-only-on-targeted-steps.
- **Simplified**: hindsight analyzer is an oracle (rule-based) rather than an LLM; environment is a tiny grid world.
- **How to extend**: replace the oracle with an LLM analyzer that outputs step indices + textual feedback, and swap `MiniGridWorld` with your tool-calling / app-world style environment.
