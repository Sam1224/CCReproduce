# IAR reproduction

This folder implements a compact PyTorch reproduction of the paper's Inject → Align → Recover pipeline for retrieval-free document knowledge internalization.

Files:

- `dataset.py`: toy creator-governance and e-commerce policy documents plus QA pairs.
- `model.py`: a small GRU-based language model proxy and a post-hoc weight merging routine for Recover.
- `train.py`: runs Inject objectives, Align QA supervision, then Recover model merging.
- `test.py`: loads the recovered checkpoint and runs toy retrieval-free QA probes.

Run:

```bash
python train.py
python test.py
```

The implementation preserves the paper's modular training logic while using a lightweight model and toy corpus so the full pipeline can be executed without proprietary corpora or large GPU resources.
