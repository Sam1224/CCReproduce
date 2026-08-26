# NativeMultimodalCTR

Toy PyTorch reproduction for **Native Multimodal Representation Learning for Click-Through Rate Prediction in E-Commerce Scenarios**.

## What is implemented

This folder provides a compact, fully synthetic reproduction of the paper's core idea: learn native multimodal item representations before CTR prediction.

- `data.py`: creates synthetic e-commerce users/items, multimodal item features, click logs, train/val/test splits, and mined triplets.
- `model.py`: implements a text-image multimodal encoder, triplet-loss representation learning, and a CTR head.
- `train.py`: runs a mine-then-train pipeline: mined triplet pretraining followed by CTR fine-tuning; saves checkpoint and history.
- `test.py`: loads the saved checkpoint and reports AUC, logloss, accuracy, and ranking metrics.

## Run

```bash
python3 train.py
python3 test.py
```

Artifacts are written to `artifacts/` when training is executed.

## Mapping to the paper

- **Native multimodal representation** is approximated by projecting synthetic text and image features into a shared item embedding, then fusing them with a learned gate.
- **Mine-then-train** is approximated by mining clicked anchor/positive pairs and hard skipped negatives from synthetic behavior logs, then optimizing triplet margin loss.
- **CTR prediction** uses the pretrained multimodal item representation with user/category/price features in a compact MLP head.

## Not implemented

- Real e-commerce images/text, industrial-scale retrieval, or proprietary feature stores.
- Exact paper hyperparameters and production candidate generation.
