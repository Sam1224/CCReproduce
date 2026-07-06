# ShopR1 reproduction

This folder implements a compact PyTorch reproduction of the main Shop-R1 idea: simulate online-shopping behavior with a two-part objective that combines supervised rationale/action learning and a GRPO-style reward over action type, sub-action attribute/value correctness, and difficulty-aware scaling.

The implementation uses a toy shopping-behavior dataset so the model, training script, and evaluation script can run end-to-end without proprietary logs.

## Files

- `dataset.py`: toy shopping dataset, vocabulary, label spaces, and collate function.
- `model.py`: `ShopR1Policy` with text encoder, action heads, rationale decoder, supervised loss, and GRPO-style reward loss.
- `train.py`: SFT + reward optimization pipeline.
- `test.py`: exact action accuracy evaluation.

## Run

```bash
python train.py --write-toy-data --epochs 8
python test.py --checkpoint checkpoints/shop_r1.pt
```
