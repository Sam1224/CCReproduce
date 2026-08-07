# ABAgent (Toy Reproduction)

This folder provides a toy but runnable PyTorch-style reproduction of:

- **A/B Agent: A Self-Evolving Agent for Strategy Iteration in Industrial A/B Testing**
- https://arxiv.org/abs/2608.04625

The original paper focuses on industrial recommendation strategy optimization with a closed-loop agent that organizes historical experiment knowledge, retrieves relevant strategies, generates an initial proposal, and keeps improving it from online feedback. The full industrial stack, experiment platform, and LLM-based generation pipeline are not public, so this reproduction keeps the method logic while simplifying the environment into a deterministic toy simulator.

## What is implemented

### 1. Strategy experience tree

Implemented in `experience_tree.py`.

- Organizes historical records by `domain -> scenario -> stage -> objective`
- Supports path-aware retrieval and tree-distance boosting
- Mimics the paper's hierarchical knowledge organization used by Tree-RAG

### 2. Multi-path retrieval + reranking

Implemented in `retriever.py`.

- Sparse TF-IDF retrieval for explicit keyword overlap
- A lightweight dense retriever trained on request-record pairs
- Tree-boosted reranking to prefer records closer to the current strategy path

### 3. Target-aware strategy generation

Implemented in `model.py`.

- Encodes business request context and metric weights
- Uses a surrogate value network to predict core metrics and guardrails
- Produces executable strategy configs (`mechanism + params`) instead of open-ended natural-language plans

### 4. Experiment-guided self-evolution

Implemented in `test.py` with utilities in `evolution.py` and `simulator.py`.

- Runs an iterative loop: retrieve -> propose -> simulate -> compare -> refine
- Uses a paper-aligned utility function that rewards core metrics and penalizes violated guardrails
- Demonstrates consistent improvement over random search and one-shot initialization on the toy benchmark

## Data

Toy data is generated and cached by `dataset.py` into `data/` on first run.

- `toy_historical_records.jsonl`: historical strategy experiments
- `toy_requests.jsonl`: new business requests for evaluation

The schema stays aligned with the training and testing scripts.

## Quickstart

From repo root:

```bash
cd 2026-08-07/ABAgent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 train.py
python3 test.py
```

## Expected output

Training writes `outputs/ab_agent.pt`.

Testing prints per-request utility comparison among:

- `random`: random strategy search
- `init`: retrieval + initial proposal only
- `evolve`: retrieval + iterative self-evolution

A typical run should show `evolve` outperforming `random`, and usually outperforming `init` as well.

## Faithful approximations vs simplifications

Faithful parts:

- Historical knowledge organized as a hierarchical experience tree
- Multi-path retrieval with structure-aware boosting
- Utility defined from core metrics and guardrail penalties
- Closed-loop self-evolution driven by experiment feedback

Toy simplifications:

- Replaces LLM text generation with a surrogate value net + param refinement
- Uses a synthetic A/B simulator instead of a production traffic system
- Uses structured toy records instead of raw experiment reports
- Focuses on runnable strategy iteration rather than full industrial deployment

## High-level pseudocode

```text
build experience tree from historical experiments
retrieve top strategies with sparse + dense + tree-aware reranking
generate an initial executable strategy config
simulate online feedback
compute utility = core gain - lambda * guardrail violation
iteratively refine or switch strategy until the best branch stabilizes
```
