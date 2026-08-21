# PolicyGuide Reproduction

This folder contains a compact PyTorch-oriented reproduction of the core idea in **PolicyGuide: From Guarding One Action to Guiding the Whole Workflow for Policy-Compliant LLM Agents**.

The implementation keeps the paper's central decomposition:

- a policy is compiled into a workflow graph;
- dialog state persists across turns;
- a proactive verifier checks the first unsatisfied workflow node before sensitive actions;
- a lightweight neural scorer can be trained on synthetic policy traces to predict the next policy node.

## Files

- `model.py`: policy graph, state tracker, symbolic verifier, and PyTorch next-node scorer.
- `dataset.py`: toy retail/customer-service policy traces aligned with the model and training script.
- `train.py`: trains the neural scorer and writes a checkpoint.
- `test.py`: smoke tests for symbolic verification and neural forward pass.

## Quick start

```bash
python test.py
python train.py --epochs 20
```

## Scope notes

The original paper evaluates full LLM agents with external workflow safeguards on τ2-bench domains. This reproduction implements the paper's algorithmic skeleton and a compatible toy pipeline; replacing the toy dataset with τ2-bench traces would require the original benchmark environment, tool schemas, and model endpoints.
