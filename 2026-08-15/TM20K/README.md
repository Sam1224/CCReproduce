# TM20K Reproduction

This folder contains a compact PyTorch reproduction of **Teacher Retains Full Tokens, Student Merges Efficiently: TM20K for E-Commerce Sequence Modeling in Ad Recommendation**.

The implementation follows the paper's core idea: train a full-token teacher on ultra-long behavior sequences, compress the student input with token-merge strategies, and distill both logits and sequence representations from the teacher to recover performance while keeping student inference efficient.

## Files

- `tm20k.py`: model, token merge modules, distillation losses.
- `data.py`: toy e-commerce sequence dataset with compatible training/evaluation interfaces.
- `train.py`: end-to-end teacher training and student distillation pipeline.
- `test.py`: smoke tests for shape, merge ratios, and one optimization step.
- `requirements.txt`: minimal runtime dependencies.

## Quick start

```bash
pip install -r requirements.txt
python test.py
python train.py --seq-len 256 --merge-ratio 0.25 --epochs 1
```

## Notes

The original industrial system uses 20K behavior tokens and production-scale ads features. This reproduction keeps the same pipeline and formulas but uses a toy dataset so the code can run on a local CPU/GPU. To scale toward the original setting, replace `ToyEcommerceSequenceDataset` with production sequence features and increase `--seq-len` to the desired maximum sequence length.
