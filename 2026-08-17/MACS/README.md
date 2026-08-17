# MACS

Toy PyTorch reproduction of **MACS: A Hybrid Multi-Agent Framework for Reliable Conversational E-Commerce Recommendation**.

This implementation keeps the paper's minimum deployable logic:

- a **shopping agent** that accumulates session-level slots across turns;
- a **merchant agent** that performs deterministic catalog grounding with hard constraints;
- a lightweight **constraint-aware reranker** trained on synthetic catalog conversations;
- a multi-turn evaluator that checks pass rate, budget compliance, and exclusion handling.

The code is intentionally small and runnable on CPU. It does **not** reproduce the paper's proprietary benchmark or LLM prompts, but it aligns the core architecture with a toy catalog pipeline.

## Files

- `data.py`: synthetic catalog, session memory, retrieval and dataset builders
- `model.py`: slot encoder, candidate encoder, reranker and pipeline wrapper
- `train.py`: trains the reranker on grounded candidate sets
- `test.py`: runs multi-turn evaluation and constraint-compliance checks

## Run

```bash
cd CCReproduce/2026-08-17/MACS
python3 train.py
python3 test.py
```

## Mapping to the paper

- **Shopping agent** → `SessionMemory` + slot updates from conversation turns
- **Merchant agent** → `retrieve_candidates` with hard brand / budget / category filters
- **Catalog grounding** → all recommendations come from the synthetic merchant catalog
- **Reliable multi-turn behavior** → session slots persist and handle exclusion reversals
