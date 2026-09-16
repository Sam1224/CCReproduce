# RegRet (Toy Reproduction)

Toy-but-runnable reproduction for:

- **RegRet: Enhancing Region-Level Retrieval in Large Multimodal Models** (arXiv:2609.16847)

## Goal

Replicate the core method logic in a lightweight setting:

- A **Region-Aware Encoder** fuses ROI detail with selective global context.
- A **multi-stage training pipeline** first learns localized semantics, then warms up text-side semantics, and finally performs region-text contrastive retrieval training.
- The final evaluation checks whether retrieval with ROI-aware fusion beats naive region-only / global-only matching on a synthetic e-commerce-style benchmark.

This implementation does **not** reproduce the paper's large-scale REGMB benchmark or full LMM stack. Instead it provides an end-to-end runnable pipeline whose interfaces match the original story: data, model, staged training, and retrieval evaluation.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
python test.py
```

## What this reproduction keeps

- Explicit ROI + global-context inputs.
- Region-aware fusion instead of whole-image-only retrieval.
- Stage-wise training rather than a single one-shot contrastive fit.
- Retrieval metrics (`Recall@1`, `Recall@5`) plus baseline comparisons.

## What is simplified

- Region features, global context, and texts are synthetic vectors instead of image/video encoders.
- The localized-captioning and pure-text stages are approximated by supervised semantic warm-up objectives.
- REGMB is replaced by a synthetic benchmark designed to contain hard negatives that share similar background but differ in local product attributes.

## Files

- `data.py`: synthetic region retrieval benchmark.
- `model.py`: region-aware encoder, text encoder, and toy RegRet model.
- `train.py`: three-stage training and checkpoint saving.
- `test.py`: retrieval evaluation and baseline comparison.
