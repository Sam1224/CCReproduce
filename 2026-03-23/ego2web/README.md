# Ego2Web (toy reproduction)

Toy reproduction skeleton for:

- **Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos** (arXiv:2603.22529)

## Idea captured

We simulate the benchmark structure: an agent observes an egocentric “video” (here: a token sequence) and must predict the corresponding **web task / action plan**.

This toy version provides:

- Synthetic egocentric sequences (events)
- A temporal encoder baseline (GRU)
- Task classification accuracy as the metric

## Not covered

- Real video understanding
- Tool-use interaction (browser, DOM, navigation)
- Multi-step action execution / evaluation harness

## Quickstart

```bash
cd ccreproduce_repo/2026-03-23/ego2web
python3 -m pip install -r requirements.txt
python3 train.py --epochs 6
python3 test.py
```
