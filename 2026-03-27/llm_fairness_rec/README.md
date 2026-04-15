# Lightweight Fairness for LLM-Based Recommendations

This repository provides a PyTorch implementation for fairness-aware recommendations using kernelized INLP and gated MoE adapters.

## Architecture
- **Kernelized INLP**: A closed-form projector for removing demographic attribute signals.
- **Gated MoE Adapter**: Restores utility signals using attribute-specific experts.

## Usage
1. Prepare user sequence data with sensitive attributes.
2. Run `python train.py` to train the model.
3. Run `python test.py` for evaluation.
