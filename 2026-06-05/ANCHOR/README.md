# ANCHOR (toy reproduction)

- Paper: **ANCHOR: Agentic Noise Creation Framework for Human Simulation and Denoising Recommendation** (arXiv:2606.05621)
- Core idea:
  - Real-world implicit feedback is noisy, but we rarely have explicit noise labels.
  - Reframe denoising as a **Creation–Recognition** loop: *create labeled noise* via simulation, then *train a recognizer* to detect noise in real logs.

This folder implements a **toy but runnable** pipeline:

- Build a synthetic implicit-feedback dataset with 5 injected noise types:
  - misclick, curiosity, caption bias, popularity bias, position bias.
- Train a parametric **noise recognizer** (binary classifier).
- Filter interactions and train a simple MF recommender on the cleaned data.

> Notes
>
>- The paper uses a much richer agentic simulation (recommender-in-the-loop + boundary refinement).
>- This toy version focuses on the core interface: **generate noise labels → train recognizer → denoise training data**.

## Quickstart

```bash
python -m pip install -r requirements.txt

# Train recognizer + recommender
python train.py --noise-epochs 3 --rec-epochs 3 --out runs/anchor.pt

# Evaluate ranking quality (Recall@K / NDCG@K)
python evaluate.py --ckpt runs/anchor.pt --k 20
```

## Files

- `data.py`: synthetic dataset + noise injection.
- `model.py`: MF recommender + noise recognizer.
- `train.py`: train recognizer, denoise, then train recommender.
- `evaluate.py`: Recall@K / NDCG@K evaluation.
