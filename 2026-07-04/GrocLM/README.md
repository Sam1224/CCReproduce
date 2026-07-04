# GrocLM Toy Reproduction

This folder reproduces the core GrocLM logic: rebuy-cycle injection, sequential relational fine-tuning, and constrained category decoding. The toy dataset mirrors grocery histories, query context, category targets, and rebuy priors.

Run:

```bash
python train.py
python test.py
```

The production paper trains LoRA adapters on Llama 3 and applies user-specific trie constraints. This reproduction uses a compact GRU/embedding model plus a fixed category vocabulary so the full pipeline is runnable without proprietary data or LLM weights.
