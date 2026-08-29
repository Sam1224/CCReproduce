# Retrieve, Match, Escalate

This folder is a PyTorch reproduction scaffold for **Retrieve, Match, Escalate: Efficient and Accurate Product Linking at Scale** (`arXiv:2608.25037`).

## What is implemented

- A multimodal two-tower retriever over product text and image features.
- A distilled cross-encoder matcher trained from soft VLM-consensus style labels.
- An escalation gate that routes uncertain pairs to an agentic VLM resolver interface.
- A toy e-commerce product-linking dataset with aligned query/candidate text, image vectors, labels, and ambiguity scores.
- End-to-end training and evaluation scripts.

## Files

- `dataset.py` builds the toy product-linking dataset and tokenizer.
- `model.py` implements the retrieve-then-match cascade and escalation policy.
- `train.py` trains retriever and matcher jointly and saves a checkpoint.
- `test.py` evaluates the saved checkpoint.

## Quick start

```bash
python train.py --epochs 40 --output-dir runs/rme_toy
python test.py --checkpoint runs/rme_toy/checkpoint.pt
```

## Notes on fidelity

The paper's production system uses large-scale catalog data, dual-VLM consensus labeling, a production ANN retrieval layer, and an agentic multimodal VLM for ambiguous cases. This reproduction keeps the same module boundaries and training interfaces while replacing proprietary catalog data and VLM calls with a toy dataset and deterministic resolver hook, so the code can run locally and be extended with real embeddings or VLM APIs.
