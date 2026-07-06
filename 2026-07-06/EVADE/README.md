# EVADE reproduction

This folder implements a compact PyTorch reproduction of an EVADE-style e-commerce evasive-content moderation baseline. It uses a toy Chinese multimodal dataset with text, lightweight image features, explicit advertising-rule targets, and category labels.

The model follows the paper motivation: e-commerce moderation needs both multimodal evidence and rule-aware reasoning, so `EvadeModerator` fuses text/image states, attends over policy rules, predicts violated rules, and outputs the final moderation category.

## Files

- `dataset.py`: toy multimodal e-commerce moderation samples and rule/category labels.
- `model.py`: text encoder, image projection, rule-aware fusion, rule prediction, and category prediction.
- `train.py`: supervised multi-task training.
- `test.py`: category accuracy and rule explanation demo.

## Run

```bash
python train.py --epochs 10
python test.py --checkpoint checkpoints/evade.pt
```
