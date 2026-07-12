# EVADE-Bench Toy Reproduction

Toy-but-runnable PyTorch reproduction for:

- **EVADE-Bench: Multimodal Benchmark for Evaluating and Enhancing Evasive Content Detection** (SIGIR 2026, arXiv:2505.17654)

## Goal

This implementation reproduces the paper's core technical ideas in a compact setting:

- E-commerce moderation requires both **complex rule comprehension** and **true intent inference** from deliberately obfuscated text-image inputs.
- The dataset contains evasive content patterns such as word splitting, euphemisms, masked claims, and image-side cues.
- The model fuses text, image, and rule embeddings to predict violation classes.
- A multi-agent decomposition separates visual description, rule reasoning, and final coordination, mirroring the paper's finding that decoupled multimodal reasoning can improve accuracy.

The original paper releases the benchmark dataset at Hugging Face, but no full training/evaluation code was found during patrol. This folder therefore provides a faithful toy pipeline with synthetic data and aligned interfaces.

## Quickstart

```bash
pip install -r requirements.txt
python train.py --epochs 3
python test.py --checkpoint checkpoints/evade_toy.pt
```

## Files

- `dataset.py`: synthetic evasive e-commerce text-image-rule dataset.
- `model.py`: multimodal detector with text, image, rule, and fusion modules.
- `multi_agent.py`: visual describer, rule reasoner, and detection coordinator pipeline.
- `train.py`: training script for the toy detector.
- `test.py`: evaluation script with full accuracy, partial accuracy, and multi-agent accuracy.

## Mapping to paper

The full EVADE-Bench contains 2,833 text samples and 13,961 images over six violation categories. This reproduction uses generated toy samples but keeps the same interface shape: text content, image features, rule category, violation label, and evasive pattern. The real benchmark can be plugged in by replacing `SyntheticEvasiveContentDataset` with a loader that returns the same keys.
