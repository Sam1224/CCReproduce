# MSA (toy reproduction)

This folder is a **lightweight, runnable reproduction skeleton** of the core idea in:

- **MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens** (arXiv:2603.23516)

## What this reproduction covers

The paper proposes an end-to-end trainable long-term memory architecture that scales to extremely large memory banks via:

- **Document-based sparse attention**: route a query to Top-k relevant documents/chunks using a learned routing projection, then attend only to the selected memory.
- **Chunk-wise pooling**: compress per-document KV into chunk representations for scalable retrieval.
- **(Paper) document-wise RoPE + inference optimizations**: out of scope in this toy.

This toy reproduction implements the key *mechanism-level* idea (routing + sparse attention over chunk-pooled memory) on a synthetic QA task:

- A memory bank contains many documents; one document contains the “needle” key-value pair.
- The query asks for the value associated with the key.
- The model learns a differentiable routing score to select Top-k documents, then answers via attention over only those documents.

## Quickstart

```bash
cd 2026-03-28/msa
python3 -m pip install -r requirements.txt
python3 train.py
python3 test.py
```

## Files

- `dataset.py`: synthetic memory QA data generator
- `model.py`: simplified MSA routing + sparse attention layer
- `train.py`: training loop
- `test.py`: smoke test

## Notes / limitations

- This is **not** a full NeurIPS-scale reproduction (no 100M-token engineering, no optimized kernels, no document-wise RoPE, no memory-parallel inference).
- It is intended as a clean reference for the *end-to-end differentiable routing + sparse attention* pattern.
