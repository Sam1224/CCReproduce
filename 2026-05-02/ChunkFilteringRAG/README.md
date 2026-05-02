# ChunkFilteringRAG (Reproduction)

This folder reproduces (in a **toy but end-to-end** way) the key idea of:

- **Reducing Redundancy in Retrieval-Augmented Generation through Chunk Filtering** (arXiv:2604.24334)

The paper studies how simple **pre-index chunk filtering** (e.g. entity-based filtering) can reduce a vector index size while keeping retrieval quality close to the baseline.

## What is reproduced here

- A minimal RAG-style retrieval pipeline with:
  - Toy corpus generation (with intentionally redundant chunks caused by overlap)
  - A small **PyTorch bi-encoder** for query/chunk embeddings
  - Retrieval evaluation with **token-level precision/recall/IoU** (similar spirit to the paper)
  - A chunk filtering step (exact-dedup + entity-set Jaccard)

This repo is not meant to reproduce the paper's exact datasets; it provides a **code-complete pipeline** that matches the method abstraction and is easy to swap with real corpora.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run baseline vs filtered comparison
python run_experiment.py
```

You should see:
- Index size reduction (filtered vs baseline)
- Retrieval precision/recall/IoU changes

## Files

- `toy_data.py`: toy corpus + query generation
- `chunking.py`: overlap chunking (introduces redundancy)
- `filtering.py`: chunk filtering implementations
- `model.py`: PyTorch bi-encoder
- `train.py`: contrastive training loop
- `eval_retrieval.py`: retrieval + token metrics
- `run_experiment.py`: end-to-end runner

## Notes

- The paper also explores semantic/topic-based filtering; this toy reproduction focuses on the most actionable variant: **entity-based filtering**.
- Replace `toy_data.py` with your own corpora to validate on real e-commerce content and governance knowledge bases.
