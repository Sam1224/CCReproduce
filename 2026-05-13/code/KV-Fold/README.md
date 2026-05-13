# KV-Fold (Reproduction)

This folder is a lightweight, runnable reproduction of **KV-Fold: One-Step KV-Cache Recurrence for Long-Context Inference** (arXiv:2605.12471).

The paper proposes a *training-free* long-context inference protocol: process a long sequence in chunks, and treat the **KV cache** as a recurrent state that is carried forward (a left-fold over chunks). This reproduction focuses on implementing that **KV-fold prefill** and providing small sanity checks.

## What is implemented

- **Chunked prefill with full KV accumulation** (`kv_fold/kv_fold.py`)
  - Forward a long prompt in chunks while carrying `past_key_values`.
  - Collect per-token logits and verify equivalence with a single full forward pass (when the model can fit the full sequence).
- **Toy needle task generator** (`kv_fold/tasks/needle.py`)
  - Generates a long text with an inserted key-value “needle” and a question at the end.
- **Scripts**
  - `scripts/eval_equivalence.py`: compares KV-Fold logits vs full forward logits.
  - `scripts/eval_needle.py`: runs a tiny end-to-end retrieval demo (configurable).

## Quickstart

```bash
cd code/KV-Fold
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Verify KV-Fold == full forward (tiny model)
python scripts/eval_equivalence.py

# 2) Run a toy needle demo
python scripts/eval_needle.py --model sshleifer/tiny-gpt2 --target-tokens 1024 --chunk-size 128
```

## Notes / Differences vs. the paper

- The original paper reports 100% needle retrieval up to 128K tokens on Llama-3.1-8B. This repo keeps the **interfaces and KV-fold logic** but uses **small default models** so the scripts are runnable on a CPU-only environment.
- The core claim we validate here is **functional equivalence**: chunked KV-fold prefill produces the same logits as a single full forward pass (up to numerical tolerance) for models that support `past_key_values`.
