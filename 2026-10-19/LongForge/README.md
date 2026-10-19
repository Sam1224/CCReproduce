# LongForge: Evidence–Assertion Reasoning Graphs for Long-Context Data

This is a self-contained reproduction scaffold for **LongForge: Measuring and Improving Long-Context Instruction Data via Evidence–Assertion Reasoning Graphs**.

The paper uses model calls to reconstruct reasoning graphs. This implementation keeps the same measurable structure and provides a PyTorch support scorer for toy data:

- split documents into sentence-indexed evidence spans;
- represent answer sentences as assertions;
- build evidence→assertion reasoning steps;
- compute the four LongForge dimensions: breadth, depth, correctness, completeness;
- synthesize/keep samples by correctness and completeness and mix by breadth/depth.

## Files

- `data.py` — toy long-context QA samples.
- `model.py` — support scorer, graph objects, analyzer, synthesizer.
- `train.py` — trains the support scorer and reports LongForge metrics.
- `test.py` — smoke test.

## Run

```bash
pip install torch
python train.py
python test.py
```

## Extension points

Replace `GraphAnalyzer.extract_evidence` and `GraphAnalyzer.judge_support` with LLM calls to match the full paper. The graph and metric APIs are stable.
