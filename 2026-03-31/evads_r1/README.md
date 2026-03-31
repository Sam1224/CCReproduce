# E-VAds-R1 (toy reproduction)

Toy reproduction skeleton for:

- **E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs** (arXiv:2602.08355)

## Idea captured

This repo provides a *lightweight* stand-in for the paper’s benchmark + RL specialization:

- Synthetic **e-commerce short video** samples with multi-modal density signals (vision/audio/OCR).
- Five task families: BP / CM / ML / CI / RC.
- A small policy model trained with a simplified **MG-GRPO** (multi-grained rewards + group relative policy optimization) to mimic the paper’s RL specialization.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-31/evads_r1
python3 -m pip install -r requirements.txt
python3 train.py --epochs 5 --rl_steps 400
python3 test.py
```

## Notes

- This is **not** a video foundation model; it uses tiny MLP encoders on synthetic features.
- The goal is to reproduce *pipeline shape*: benchmark dataset → supervised warmup → RL specialization.
