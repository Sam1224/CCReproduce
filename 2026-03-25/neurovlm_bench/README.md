# NeuroVLM-Bench (toy reproduction)

Toy reproduction skeleton for:

- **NeuroVLM-Bench: Evaluation of Vision-Enabled Large Language Models for Clinical ...** (arXiv:2603.24846)

## Idea captured

We build a tiny “clinical VLM benchmark” as multi-task classification:

- Input is an "image" feature vector (synthetic) plus a question type.
- Output is the answer label.
- We report accuracy by task type.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-25/neurovlm_bench
python3 -m pip install -r requirements.txt
python3 train.py --epochs 8
python3 test.py
```
