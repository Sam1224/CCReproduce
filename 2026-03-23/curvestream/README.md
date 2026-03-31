# CurveStream (toy reproduction)

Paper: **CurveStream: Boosting Streaming Video Understanding in MLLMs via Curvature-Aware Hierarchical Visual Memory Management** (arXiv:2603.19571)

This folder provides a **toy PyTorch reproduction** of the paper’s *core systems idea*: in a streaming setting, use **curvature-aware hierarchical memory** to keep a small set of informative frames.

## What is implemented

- A synthetic “streaming video feature” dataset where a sequence contains an **event** (direction change) that creates **high curvature** in the feature trajectory.
- A simple model that predicts the event class using only a small **memory** of frames.
- A **curvature-aware** memory policy that keeps (1) high-curvature frames as long-term memory and (2) the most recent frames as short-term memory.

## What is NOT implemented

- No real video frames, VLM/MLLM backbone, or true streaming benchmark.
- No full memory update / retrieval design from the original paper.

Where the original paper has more complex components (multi-level visual memory updates, MLLM integration, and benchmark-specific engineering), this toy repo keeps the interface and the selection logic, and leaves the rest as a clearly-scoped TODO for real reproduction.

## Quickstart

```bash
pip install -r requirements.txt
python3 train.py --policy curvature
python3 test.py
```

To compare against baselines:

```bash
python3 train.py --policy uniform
python3 train.py --policy recent
```

The checkpoint is saved to `checkpoints/curvestream.pt`.
