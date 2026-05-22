# STAL (Spectral Tail Auxiliary Learning) — Toy Reproduction

This folder is a minimal **PyTorch** reproduction scaffold for the idea in:

**"Spectral Tail Auxiliary Learning for AI-Generated Image Detection"** (arXiv:2605.22751)

The paper studies a robust frequency-domain cue (*spectral tail uplift*) and proposes **STAL**:

- train a **frequency teacher** that captures tail cues from spectra
- distill its representation to a **spatial detector** during training
- **discard all frequency modules at inference** (no extra inference overhead)

This reproduction implements the same *training-time-only frequency supervision* pattern with a toy dataset.

## Toy dataset

- "real" images: smooth low-frequency noise
- "fake" images: real + injected ultra-high-frequency tail energy (simulating tail uplift)

## Files

- `dataset.py`: synthetic dataset + spectrum feature extraction (radial log-power)
- `model.py`: frequency teacher + spatial detector
- `train.py`: joint training with STAL losses
- `test.py`: inference with *spatial detector only*

## Quickstart

```bash
cd CCReproduce/2026-05-22/stal
python3 train.py --epochs 5
python3 test.py --ckpt checkpoints/spatial.pt
```

## Notes

- This is a **toy** faithful pattern reproduction: it preserves the *architecture of supervision* (frequency teacher → distill → discard at inference).
- To plug in real datasets, replace `ToyAIGenDataset` with an image-folder dataset, and keep `compute_radial_spectrum()`.
