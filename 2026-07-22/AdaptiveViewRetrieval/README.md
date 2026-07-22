# Adaptive View Retrieval

Toy but runnable PyTorch reproduction of **Now You See the Hate: Adaptive View Retrieval for Hidden Hateful Illusions** (arXiv:2607.19061).

This implementation follows the paper's core idea: hidden harmful content is recovered first as a **multi-view template retrieval** problem, then calibrated into a harmful/benign moderation decision. The real paper uses frozen CLIP features over perceptual view banks. This reproduction keeps the same interfaces and training objectives with a lightweight frozen CNN so it can run on CPU with synthetic hidden-shape data.

## Files

- `dataset.py`: synthetic illusion-like dataset and template bank.
- `model.py`: perceptual view bank, frozen encoder, adaptive gate, retrieval head, calibration head, and loss.
- `train.py`: end-to-end training pipeline on the toy dataset.
- `test.py`: checkpoint evaluation with balanced accuracy and retrieval accuracy.

## Quick start

```bash
python train.py --epochs 3 --output avr_toy.pt
python test.py --checkpoint avr_toy.pt
```

## Mapping to the paper

The implementation includes the main mechanisms from the paper: a complementary view bank, shared feature extraction for images and hidden-message templates, adaptive per-sample view weights, template retrieval, a harmfulness calibration head, and the combined training objective of retrieval cross-entropy, binary calibration loss, and gate entropy regularization. Dataset-specific optical transforms and CLIP/open benchmark evaluation are simplified because the original benchmark and moderation label space are not public in this environment.
