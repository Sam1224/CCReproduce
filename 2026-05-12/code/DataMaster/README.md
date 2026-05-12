# DataMaster (Reproduction)

This folder is a lightweight, runnable reproduction of **DataMaster: Towards Autonomous Data Engineering for Machine Learning** (arXiv:2605.10906).

> Note: The paper links to an "official code" repository, but at the time of this reproduction that repository only contains a placeholder README ("Code coming soon"). Therefore, this reproduction re-implements the core *framework idea* (DataTree + shared DataPool + GlobalMemory + downstream-feedback evaluation) with a toy, self-contained task.

## What is implemented

- **DataTree**: a tree-structured search over *data states*.
- **Red nodes**: external data discovery / composition actions (add a new source from a shared **DataPool**).
- **Black nodes**: data cleaning / transformation actions applied to the current composed dataset.
- **GlobalMemory**: records node actions, scores, and artifacts for reuse and to avoid repeated failures.
- **Downstream feedback loop**: each node is validated by training a *fixed* model (a small PyTorch text classifier) and evaluating on a held-out validation split.

## Quickstart

```bash
cd code/DataMaster
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the toy DataMaster search
python scripts/run_search.py

# Train/eval a single data state (baseline)
python scripts/train_eval.py --recipe baseline
```

## Output

- The search script prints the best data recipe found and its validation metrics.
- It also saves a JSON trace under `artifacts/` (search tree + memory).

## Mapping to the paper

This reproduction focuses on the framework components emphasized in the paper:

- **DataTree** (tree-structured branching over data-engineering options)
- **Shared Data Pool** (reusable discovered sources)
- **Global Memory** (reuse across branches)

The LLM-driven online dataset discovery in the paper is replaced here by a deterministic toy `DataPool` with multiple pre-defined sources, so the full pipeline is runnable locally.
