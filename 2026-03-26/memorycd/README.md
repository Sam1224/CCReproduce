# MemoryCD (toy reproduction)

Toy reproduction skeleton for:

- **MemoryCD: Benchmarking Long-Context User Memory of LLM Agents for Lifelong Cross-...** (arXiv:2603.25973)

## Idea captured

We simulate “long-context user memory” as a key-value recall benchmark:

- Each episode contains a set of user facts (preferences).
- A query asks for one fact.
- A memory-aware model attends over stored facts to answer.

We report recall@1 / accuracy.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-26/memorycd
python3 -m pip install -r requirements.txt
python3 train.py --epochs 8
python3 test.py
```
