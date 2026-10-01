# Ranking-PE toy reproduction

This folder provides a compact PyTorch reproduction of **Ranking-Aware Prompt Optimization for Multimodal Clinical Diagnosis** in an e-commerce creator-governance setting.

The original paper changes reflective prompt evolution from instance-level correctness to positive-negative pair ordering, so candidate prompts are selected by empirical AUROC rather than raw accuracy. This is important for rare positive classes, where high accuracy can be achieved by a conservative majority-class prompt.

## Files

- `data.py`: imbalanced synthetic multimodal governance dataset with text, image and creator metadata features.
- `model.py`: frozen MLLM-style risk scorer plus prompt candidates represented as feature priors.
- `train.py`: trains the toy scorer, runs Accuracy-PE and Ranking-PE prompt selection, and writes `train_metrics.json`.
- `test.py`: reloads the checkpoint and writes `test_metrics.json`.

## Run

```bash
python train.py
python test.py
```

The toy setup intentionally keeps the dataset small and CPU-friendly while preserving the key mechanism: prompt candidates are evaluated through either a correctness-score matrix or a pair-ordering-score matrix, and the latter is expected to improve AUROC on imbalanced governance data.
