# TM20K (Toy Reproduction)

Toy-but-runnable PyTorch reproduction for:

- **Teacher Retains Full Tokens, Student Merges Efficiently: TM20K for E-Commerce Sequence Modeling in Ad Recommendation** (arXiv:2608.07055)

## What is reproduced

This implementation keeps the paper's main recipe in a compact industrial-toy setting:

- a **full-token teacher** trained on long e-commerce behavior sequences;
- a **token-merged student** that preserves recent tokens while compressing older history with three merge views inspired by **LITM / PATM / LPTM**;
- a **knowledge distillation** stage from the teacher to the student;
- a **truncated baseline** that only looks at the latest interactions.

The synthetic task is binary ad response prediction on long user histories. The label depends on both **recent intent** and **older long-term support**, so a last-window baseline loses information while TM20K-style compression remains effective.

## Files

- `data.py`: long-sequence synthetic e-commerce ad-response data and AUC metric.
- `model.py`: teacher, token-merged student, and truncated baseline.
- `train.py`: supervised teacher/baseline training + teacher-to-student distillation.
- `test.py`: checkpoint loading and final AUC/latency comparison.

## Quickstart

```bash
python train.py
python test.py
```

## Expected outcome

You should observe the same qualitative pattern as the paper:

- full teacher performs best,
- token-merged student recovers part of the long-range signal relative to a short-window baseline,
- naive truncation drops useful historical evidence,
- the toy implementation focuses on the algorithmic idea rather than production-grade kernel optimization.
