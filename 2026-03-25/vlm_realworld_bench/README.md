# VLM RealWorld Bench (toy reproduction)

Toy reproduction skeleton for:

- **How Far Are Vision-Language Models from Constructing the Real World? A Benchmark ...** (arXiv:2603.24866)

## Idea captured

We emulate a “world construction” benchmark as **scene reasoning**:

- A synthetic scene encodes objects and their 2D positions (as an "image" feature vector).
- A question asks about relations (left/right/closest/count).
- A VLM-style fusion model predicts the answer.
- We report **in-domain** and **OOD** accuracy (shifted coordinate distribution) to reflect robustness.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-25/vlm_realworld_bench
python3 -m pip install -r requirements.txt
python3 train.py --epochs 10
python3 test.py
```
