# EvoReason (toy reproduction)

This folder provides a runnable PyTorch toy reproduction for **EvoReason: Self-Evolving Reasoning Primitive-Guided On-Policy Distillation for Latent Reasoning in Generative Recommendation**.

## What is preserved
- reasoning primitive library and primitive-guided teacher signals
- latent reasoning tokens distilled from explicit reasoning traces
- on-policy refinement that lets the student update its latent reasoning path during recommendation training

## What is simplified
- synthetic creator-content recommendation data
- compact GRU-based history encoder and lightweight latent reasoner
- teacher trajectories generated from heuristic primitives instead of a production LLM agent

## Run
```bash
python train.py --epochs 6
python test.py --checkpoint evoreason_toy.pt
```
