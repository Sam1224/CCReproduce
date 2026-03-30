# Self-Organizing Multi-Agent Systems (toy reproduction)

Toy reproduction skeleton for:

- **Self-Organizing Multi-Agent Systems for Continuous Software Development** (arXiv:2603.25928)

## Idea captured

We simulate continuous development as a stream of tasks (bugfix/feature/refactor) and a pool of agents with different strengths.

A small learned **router** assigns tasks to agents to maximize throughput/quality. This captures the “self-organizing” aspect via online learning.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-26/self_organizing_agents
python3 -m pip install -r requirements.txt
python3 train.py --episodes 2000
python3 test.py
```
