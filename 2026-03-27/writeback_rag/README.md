# writeback_rag (toy reproduction)

This folder is a **lightweight, runnable reproduction** of the core idea in **WRITEBACK-RAG** (arXiv:2603.25737): *treat the external RAG corpus as a trainable asset, and “write back” distilled evidence units into the knowledge base as an offline preprocessing step*.

It is not a full-scale reproduction of the paper’s LLM distillation setup (their experiments rely on large corpora + strong LLMs + multiple RAG pipelines). Instead, it implements a **faithful pipeline skeleton** with a **toy dataset** that demonstrates the mechanism end-to-end:

- Baseline RAG retrieves from a raw corpus.
- A gating stage chooses samples that benefit from retrieval.
- A document gate selects contributing documents.
- A distiller fuses/compresses evidence into a compact knowledge unit.
- The knowledge unit is written back and indexed alongside the original corpus.

## Quickstart

```bash
cd 2026-03-27/writeback_rag
python3 -m pip install -r requirements.txt
python3 main.py
```

Expected output is something like:

- Baseline accuracy: X / N
- Writeback accuracy: Y / N

## How it maps to the paper

- **Utility gate**: identifies samples that are helped by retrieval.
- **Document gate**: keeps only retrieved docs that contain the answer evidence.
- **Distiller**: in the paper this is an LLM distiller; here we implement an extractive distiller that fuses answer-bearing sentences (and include a TODO/pseudocode hook for plugging in a summarization model).
- **Write-back**: the distilled unit is added as a new “document” and indexed.

## Files

- `toy_data/corpus.jsonl`: raw corpus (each line is `{id, text}`)
- `toy_data/qa.jsonl`: evaluation QA (each line is `{id, question, answer}`)
- `main.py`: end-to-end demo pipeline
- `src/`: minimal components (embedding retriever, gates, distiller)

## Limitations (explicit)

- The generator is a toy extractive QA heuristic.
- Distillation is extractive (not LLM-based), but the code is structured so you can replace it with a real distiller.
- The dataset is tiny by design; the goal is to provide a clean and runnable reference implementation.
