# Positional Task Conditioning for Product-Catalog Defect Detection

PyTorch reproduction scaffold for **Positional task conditioning for scalable defect detection across product families in large product catalogs** (`arXiv:2609.09567v2`).

This implementation captures the paper's core deployable recipe:

- product-family inputs shared across several focused defect subtasks;
- explicit task conditioning at every item position;
- a single student model trained with supervised labels plus teacher-distribution distillation;
- toy data for duplicate variants, unit mismatch, theme defect, and attribute overstuffing;
- end-to-end train and test scripts.

The original Amazon Catalog AI data and prompts are proprietary, so the synthetic dataset keeps the same tensor contracts and task decomposition while avoiding private data.

```bash
python train.py
python test.py
```
