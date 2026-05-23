# CEDAR (Conceptual Embedding Disentanglement via Adaptive Rotation) — Toy Reproduction

This folder is a minimal **PyTorch** reproduction scaffold for the idea in:

**"Conceptualizing Embeddings: Sparse Disentanglement for Vision–Language Models"** (CEDAR, arXiv:2605.22679)

The paper proposes a **post-hoc, invertible, dimension-preserving** reparameterization of a frozen embedding space:

- learn an **orthogonal / invertible transform** `U`
- in the transformed space, keep only **top-k** coordinates (sparsity bottleneck)
- reconstruct back with `U^{-1}`

Compared to sparse autoencoders (SAEs), CEDAR aims to **avoid expanding the representation dimension**, preserving the original geometry.

## What this toy reproduction keeps faithful

- **Change-of-basis + top-k** sparsification (training-time bottleneck)
- **Invertibility** (we use an orthogonal matrix so `U^{-1} = U^T`)
- **Reconstruction objective** on frozen embeddings (no backbone fine-tuning)

## Toy data

We generate embeddings from a known compositional latent structure:

- factors: `color` (4) × `shape` (4) × `style` (3)
- true factors are linearly mixed into a D-dim embedding
- goal: learn a rotation where each sample activates only a few coordinates (top-k) while still reconstructing well

## Files

- `dataset.py`: synthetic factorized embeddings
- `model.py`: CEDAR rotation + top-k bottleneck
- `train.py`: trains `U` with L1 reconstruction loss
- `test.py`: evaluates reconstruction/sparsity + shows factor-axis alignment

## Quickstart

```bash
cd CCReproduce/2026-05-23/cedar
python3 train.py --epochs 200
python3 test.py --ckpt checkpoints/cedar.pt
```

## Notes

- This is a **toy but runnable** implementation focusing on the core CEDAR mechanism.
- To adapt to real CLIP/BLIP embeddings, replace `ToyEmbeddingDataset` with a dataloader that yields frozen embeddings `z` and keep the same `CEDAR` module.
