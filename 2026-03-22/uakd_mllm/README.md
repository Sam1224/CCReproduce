# UAKD-MLLM (toy reproduction)

Toy, runnable reproduction skeleton for:

- **Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models** (arXiv:2603.21426)

## Idea captured

We simulate *uncertainty-aware distillation* in a multimodal classifier setting:

- Train a **teacher** multimodal model.
- Distill to a **student** via KL (logit matching).
- Use teacher predictive **uncertainty** (entropy) to down-weight uncertain samples during distillation.

This matches the high-level mechanism: uncertainty-aware weighting makes distillation less sensitive to ambiguous/noisy samples.

## What is NOT covered

- Real MLLM architectures (vision encoder + LLM)
- Token-level generation distillation
- Large-scale multimodal datasets

## Quickstart

```bash
cd ccreproduce_repo/2026-03-22/uakd_mllm
python3 -m pip install -r requirements.txt

python3 train.py --epochs-teacher 3 --epochs-student 5
python3 test.py
```

Expected output prints student accuracy for **vanilla KD** vs **uncertainty-aware KD**.
