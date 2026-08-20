# ReWEIGH (Toy Reproduction)

This folder is a lightweight PyTorch reproduction of the core idea in:

- **ReWEIGH the Evidence: Calibrating Token-Level Ordinal Visual Evidence to Mitigate Hallucinations in Large Vision-Language Models**
- arXiv: https://arxiv.org/abs/2608.19075

The original paper proposes a training-free decoding intervention for LVLM hallucination mitigation. This reproduction focuses on the algorithmic mechanism rather than reproducing the full LVLM benchmark stack.

## What is implemented

`reweigh.py` implements the main ReWEIGH components: dense mean reciprocal rank evidence from visual-token vocabulary readouts, token-specific reference calibration from unlabeled images, stable-token registration, and bounded logit penalties for candidates with weak image evidence.

`model.py` provides a small LVLM-like toy model with visual tokens, text context, and a shared vocabulary projection head so the ReWEIGH processor can be run end to end.

`dataset.py` builds a toy e-commerce captioning dataset with product objects and attributes. It is intentionally small but keeps the same interface expected by the calibration and evaluation scripts.

`train.py` performs the paper's training-free calibration stage and saves token-level reference evidence.

`test.py` compares greedy decoding with and without ReWEIGH and reports the object-token hallucination rate.

## Quickstart

```bash
cd 2026-08-20/ReWEIGH
python3 train.py --samples 128
python3 test.py --samples 64
```

Expected output includes the number of registered tokens and baseline vs. ReWEIGH hallucination rates on the toy data.

## Notes / gaps vs. the original paper

This reproduction does not include a full 7B-32B LVLM backbone, COCO/POPE/MME/amber-style evaluation, or production image preprocessing. The implemented pipeline preserves the key method logic: calibrate token-specific ordinal visual evidence on unlabeled images, cache per-image evidence at inference, and penalize visually unsupported candidate tokens with minimal decoding overhead.
