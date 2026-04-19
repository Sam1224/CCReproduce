# GenRec (Toy Reproduction)

Paper: **GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation** (arXiv:2604.14878)

This is a *toy* PyTorch reproduction that mirrors the paper’s key ideas in a minimal, runnable setup:

- **Semantic IDs**: each item is represented by a small multi-token Semantic-ID sequence.
- **PW-NTP (page-wise supervision)**: trains the model to generate a whole recommendation page (a list) instead of a single next item.
- **Token Merger (prefill compression)**: compresses the *history/prefix* token embeddings before the transformer to reduce prompt length.
- **GRPO-style preference alignment (toy)**: samples multiple candidate pages, computes a synthetic reward, and applies a group-relative policy gradient with an NLL regularizer.

This code is intended as an implementation reference and a runnable scaffold; it does not attempt to match the paper’s industrial-scale data or exact architectures.

## Quickstart

```bash
pip install torch

# supervised pretraining (PW-NTP)
python train.py

# evaluation (Hit@K on a synthetic next-page task)
python test.py

# optional: toy GRPO-style alignment
python rl_train.py
```

## Files

- `dataset.py`: synthetic user/item interactions + Semantic-ID tokenizer
- `model.py`: causal transformer + prefix TokenMerger
- `train.py`: PW-NTP training
- `rl_train.py`: GRPO-style alignment (toy)
- `test.py`: simple ranking metrics
