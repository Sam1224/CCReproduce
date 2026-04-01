# Simula (toy reproduction)

Paper: **Reasoning-Driven Synthetic Data Generation and Evaluation** (arXiv:2603.29791)

This is a minimal end-to-end pipeline showing the *shape* of a reasoning-driven synthetic data workflow:

- generate structured synthetic tasks + reasoning traces
- train a model on the synthetic data
- evaluate on an in-distribution split and a shifted “hard” split

## Run

```bash
python3 train.py --epochs 5
python3 test.py --ckpt ckpt.pt
```

## Notes

- This toy reproduction does **not** call LLMs to generate data; it uses deterministic templates to keep it runnable.
- The goal is to provide a code scaffold (data/model/train/test) for later replacement with real LLM-based generation and evaluation.
