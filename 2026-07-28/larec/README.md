# LaRec

Toy but runnable PyTorch reproduction of **LaRec: Unleashing LLM-based Latent Reasoning for Generative Recommendation**.

## What is implemented
- **Latent Pre-training**: builds multi-step latent states from user history and aligns them with explicit preference steps.
- **Step-level Alignment**: matches each latent step to a supervision step embedding.
- **Process Direction Alignment**: constrains latent transitions to move toward the target item semantics.
- **Personalized RL-tuning**: samples user-specific latent start states from interest prototypes and optimizes recommendation rewards.
- **End-to-end pipeline**: toy dataset, model, training script, evaluation script, and checkpoint saving.

## Simplifications
- The original paper uses a much larger LLM backbone, industrial features, and GRPO-style optimization. This reproduction keeps the paper's two-stage logic but replaces the backbone with lightweight item/history encoders and a compact policy-style tuning objective so the code stays runnable on commodity hardware.
- The toy dataset preserves the same interfaces needed by the train/test scripts: user history, latent supervision steps, target item, user interest prototypes, and negative samples.

## Files
- `toy_data.json`: small recommendation dataset.
- `dataset.py`: dataset loader and collator.
- `model.py`: LaRec modules.
- `train.py`: two-stage training (latent pre-train + personalized RL-tuning).
- `test.py`: HR/NDCG evaluation and latency report.

## Usage
```bash
python train.py
python test.py
```
