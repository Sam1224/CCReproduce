# ReLog (Toy Reproduction)

Paper: **Logging Like Humans for LLMs: Rethinking Logging via Execution and Runtime Feedback** (arXiv:2603.29122)

This reproduction is a **minimal runnable proxy** of the paper’s core idea:
**iteratively refining logging statements using runtime feedback**.

Because the original work is an LLM-driven software engineering framework (not a neural model to train from scratch), this folder provides:

- a tiny toy dataset of (code, runtime trace) → (log level, log template)
- a lightweight PyTorch model that selects a template / level
- a runtime-feedback “refiner” that chooses among top-k templates using trace-aware scoring

## Quickstart

```bash
pip install torch

# Train a tiny selector model
python3 train.py --out ckpt.pt

# Run the refinement loop and report accuracy
python3 test.py --ckpt ckpt.pt
```

## What is implemented vs. simplified

Implemented (toy):
- execution-aware refinement loop (choose the best template using runtime trace signals)
- a small learnable component for template/level selection

Simplified:
- the paper uses LLMs for generation/evaluation/repair; here we replace them with a small classifier + heuristic evaluation
- compilation repair is not applicable to the toy Python snippets and is not modeled

