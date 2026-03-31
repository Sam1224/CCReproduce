# memma (toy reproduction)

Toy, runnable reproduction skeleton for:

- **MemMA: Coordinating the Memory Cycle through Multi-Agent Reasoning and In-Situ Self-Evolution** (arXiv:2603.18718)

## What this reproduction captures

MemMA’s core idea is to treat *memory construction, retrieval, and utilization* as a coordinated **memory cycle** rather than isolated heuristics, and to add a **backward path** that converts downstream failures into **memory repair actions** (in-situ self-evolution).

This toy implementation simulates the 3 roles:

- **MetaThinker**: decides what to store (a simple policy over facts)
- **MemoryManager**: stores and retrieves key-value memories via embedding similarity
- **QueryReasoner**: answers queries using retrieved memories
- **SelfEvolver**: probes the memory with synthetic QA checks and repairs missing items

No LLM is used here; it is a mechanism-level simulation that demonstrates the benefit of a repair loop.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-29/memma
python3 -m pip install -r requirements.txt
python3 train.py
python3 test.py
```

## Files

- `dataset.py`: synthetic episodic memory task
- `agent.py`: MetaThinker / MemoryManager / QueryReasoner / SelfEvolver
- `train.py`: compares w/ and w/o self-evolution
- `test.py`: smoke test
