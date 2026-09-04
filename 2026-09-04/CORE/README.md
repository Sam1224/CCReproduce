# CORE toy reproduction

This folder provides a **toy but runnable** PyTorch reproduction of **CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation**.

## What is implemented
- Five-level compositional candidate synthesis (`L5` full match to `L1` total error)
- Embedding model with query encoder + candidate encoder
- Reranker-style teacher supervision approximated by graded teacher scores
- Listwise `Rank-KL` distillation into the embedding space
- Synthetic retrieval evaluation with Top-1 / MRR metrics

## What is simplified
- Images are represented by synthetic attribute-object-relation vectors instead of real visual features.
- The reranker teacher is represented by graded teacher scores rather than a full cross-attention MLLM reranker.
- Benchmarks such as COLA / SUGARCREPE++ / NEGBENCH are replaced by a synthetic compositional retrieval benchmark that preserves the paper's supervision shape.

## Files
- `data.py`: synthetic compositional data and candidate-list generation
- `model.py`: CORE embedding model and Rank-KL objective
- `train.py`: trains the toy model and saves `core_model.pt`
- `test.py`: evaluates retrieval quality on held-out synthetic examples

## Run
```bash
python train.py
python test.py
```

## Expected outcome
The toy model should learn to rank the `L5` full-match candidate above progressively noisier candidates, illustrating the paper's central idea: **distill fine-grained reranker judgments into an embedding model**.
