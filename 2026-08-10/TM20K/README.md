# TM20K Reproduction

This folder implements a compact PyTorch reproduction of **Teacher Retains Full Tokens, Student Merges Efficiently: TM20K for E-Commerce Sequence Modeling in Ad Recommendation**.

The implementation follows the paper's core pipeline: a full-attention teacher consumes the complete behavior sequence, a student compresses ultra-long behavior tokens with token merging, and the student is trained with a mixture of hard labels and teacher-logit distillation. The included dataset is synthetic but exposes the same interfaces used by the model and training scripts: `sequence`, `target`, and `label`.

## Files

`dataset.py` builds toy e-commerce long behavior sequences with target-ad click labels. `model.py` contains the full-attention ranker, token merge modules, and distillation loss. `train.py` trains teacher then student and saves both checkpoints. `test.py` evaluates a trained checkpoint or runs a smoke test with an untrained student.

## Quick start

```bash
python train.py --epochs 2 --dataset-size 2048 --max-seq-len 512 --merged-len 128
python test.py --checkpoint checkpoints/tm20k_student.pt
```

## Mapping to the paper

The full-token teacher corresponds to the one-time heavily trained teacher in TM20K. The student uses full Transformer attention after sequence compression, rather than target-only attention. The `mean`, `recency`, and `attention` merge modes are lightweight versions of token merging strategies designed to keep fine-grained long-sequence information while reducing serving cost.

The production paper reports deployment-scale metrics such as ADSS +1.036% with only +5.6% serving latency. Those numbers require ByteDance production logs and infrastructure, so this reproduction focuses on the reproducible algorithmic skeleton and a runnable toy pipeline.
