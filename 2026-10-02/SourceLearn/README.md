# SourceLearn reproduction

PyTorch-oriented lightweight reproduction of **SourceLearn: From Knowledge Access to Source Learning**.

This implementation keeps the paper's core interface:

- persistent source model built from an authoritative source;
- self-directed source learning that revisits uncertain entities;
- task-guided source learning that updates reusable source knowledge from failures;
- retrieval + answer pipeline evaluated on a toy same-source QA dataset.

The original linked repository only contained `README.md` and `LICENSE` at inspection time, so this folder provides a compact runnable reproduction.

## Files

- `sourcelearn.py`: model, memory, learner and training/evaluation logic.
- `toy_data.py`: toy source documents and QA tasks aligned with e-commerce governance/content operations.
- `train.py`: runs self-directed and task-guided learning.
- `test.py`: verifies retrieval quality and source-model updates.

## Quick start

```bash
pip install torch
python train.py
python test.py
```
