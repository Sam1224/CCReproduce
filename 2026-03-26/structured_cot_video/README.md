# Structured CoT for Video (toy reproduction)

Toy reproduction skeleton for:

- **Reinforcing Structured Chain-of-Thought for Video Understanding** (arXiv:2603.25942)

## Idea captured

We model a two-step reasoning pipeline:

1) Predict a **structured intermediate** (a “reasoning step”).
2) Predict the final answer conditioned on that step.

We then apply a lightweight **REINFORCE** stage to optimize final-answer reward while sampling the intermediate step.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-26/structured_cot_video
python3 -m pip install -r requirements.txt

# Stage1: supervised
python3 train.py --mode supervised --epochs 6

# Stage2: reinforce
python3 train.py --mode rl --epochs 6

python3 test.py
```
