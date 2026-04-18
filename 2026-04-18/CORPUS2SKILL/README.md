# CORPUS2SKILL (Reproduction Scaffold)

This folder provides a **minimal, runnable reproduction scaffold** for the paper:

- **Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills for QA and RAG**
  - https://arxiv.org/abs/2604.14572

## What is implemented

A lightweight implementation of the paper’s core idea:

- **Offline compilation**: cluster a document corpus into a hierarchical directory structure.
- **Filesystem skills**: emit `SKILL.md` / `INDEX.md` files that describe nodes in the hierarchy.
- **Serve-time navigation**: a simple agent navigates the hierarchy by matching the query against node summaries, then retrieves documents.

Because the original paper relies on LLM summarization and an LLM agent, this reproduction uses **deterministic summarization** (keyword-based) and a **non-LLM navigation policy** to keep it runnable end-to-end.

## Quickstart

Create a venv and install deps:

```bash
pip install -r requirements.txt
```

Build the skill tree ("training"):

```bash
python train.py --corpus data/toy_corpus.jsonl --out_dir outputs
```

Run the toy evaluation:

```bash
python eval.py --qa data/toy_qa.jsonl --out_dir outputs --top_k 3
```

## Outputs

After `train.py`:

- `outputs/skills/` – generated hierarchy containing `SKILL.md` and `INDEX.md`
- `outputs/docs/` – document store (markdown files)
- `outputs/index.json` – machine-readable hierarchy index
- `outputs/vectorizer.joblib` – TF-IDF vectorizer used by the agent

## Notes

- This is a **functional reproduction scaffold**, not an attempt to exactly match the paper’s LLM-based pipeline.
- The API boundaries (corpus -> skill tree -> agent navigation -> retrieval -> QA eval) are intentionally aligned with the paper’s story so you can swap in real embeddings, LLM summaries, and LLM navigation later.
