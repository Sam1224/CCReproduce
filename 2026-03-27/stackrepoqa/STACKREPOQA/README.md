# StackRepoQA (toy reproduction)

This folder is a **runnable reproduction skeleton** for:

- **Beyond Code Snippets: Benchmarking LLMs on Repository-Level Question Answering** (arXiv:2603.26567)

The paper’s core claim is that “repo-level QA” requires reasoning over **multiple files, symbols, and long-range dependencies**, beyond single-snippet tasks.

## What this reproduction covers

This reproduction provides an end-to-end, *repo-level QA* pipeline that is small but runnable:

1. **Repo indexing**: scan a target repository, chunk files, and build an in-memory corpus.
2. **Toy dataset builder**: generate repo-level QA examples from Python files (extract docstrings + function names as questions).
3. **Retriever baseline** (PyTorch bi-encoder): learn a tiny embedding model to retrieve the best supporting chunk.
4. **Answerer baseline**: return an extractive answer from the retrieved chunk.
5. **Evaluation**: exact match + token F1 on the toy dataset.

## What is NOT covered (paper-level TODOs)

- Multi-hop reasoning across multiple retrieved files
- LLM-based reasoning/answer synthesis
- Human-verified benchmark tasks and realistic difficulty buckets
- Official dataset construction and split protocol

Those parts are left as TODO notes inside the code (with clear extension points).

## Quickstart

```bash
cd ccreproduce_repo/2026-03-27/stackrepoqa/STACKREPOQA
python3 -m pip install -r requirements.txt

# Build a toy dataset from a repo (defaults to this repo)
python3 build_toy_dataset.py --repo ../../../.. --out data/toy.json

# Train retriever
python3 train.py --data data/toy.json

# Evaluate
python3 test.py --data data/toy.json
```

## Files

- `repo_index.py`: repo scanning + chunking
- `dataset.py`: dataset schema + loaders
- `model.py`: tiny PyTorch bi-encoder retriever
- `build_toy_dataset.py`: toy QA generation from Python AST
- `train.py`: train retriever on toy pairs
- `test.py`: retrieval + extractive answering + metrics
