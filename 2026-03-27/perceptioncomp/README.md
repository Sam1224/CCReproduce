# PerceptionComp (toy reproduction)

Toy reproduction skeleton for:

- **PerceptionComp: A Video Benchmark for Complex Perception-Centric Reasoning** (arXiv:2603.26653)

## Idea captured

We simulate a video reasoning benchmark where the answer depends on **temporal perception**:

- Each "video" is a sequence of frames with a few object attributes.
- The task asks about a compositional event (e.g., did a "red" object appear after a "blue" object?).
- We train a temporal encoder baseline and report accuracy.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-27/perceptioncomp
python3 -m pip install -r requirements.txt
python3 train.py --epochs 8
python3 test.py
```
