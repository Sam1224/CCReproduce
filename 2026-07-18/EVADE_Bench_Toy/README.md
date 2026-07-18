# EVADE-Bench Toy Reproduction

This folder implements a runnable PyTorch surrogate inspired by **EVADE-Bench: Chinese Multimodal Evasive Content Detection in E-Commerce Scenarios**.

The original work is primarily a benchmark for evaluating LLM/VLM robustness on evasive e-commerce violations. This reproduction preserves the core evaluation pipeline with a CPU-friendly model and toy data:

- `data.py` creates synthetic live-commerce product posts with text tokens, colored image patches, violation labels, cross-modal mismatch signals, and evasion masks.
- `model.py` implements a text evidence encoder, visual evidence encoder, cross-modal consistency head, violation/category classifiers, and evasion-region decoder.
- `train.py` jointly trains violation detection, violation-category classification, cross-modal consistency, and evasion localization.
- `test.py` loads a checkpoint and reports category accuracy, violation accuracy, consistency accuracy, and evasion-mask IoU.

Run:

```bash
pip install -r requirements.txt
python train.py --epochs 4
python test.py --checkpoint checkpoints/evade_bench_toy.pt
```

This is intentionally toy-scale. The original expert-curated Chinese text/image dataset and proprietary production moderation policy taxonomy are replaced with synthetic examples while keeping interfaces aligned with multimodal governance evaluation.
