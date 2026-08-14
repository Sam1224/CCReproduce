# LinesAndLadders toy reproduction

This folder contains a toy but runnable reproduction of **Lines and Ladders: A Context-Aware Multi-Agent Framework for Large-Scale Retail Price Taxonomy**.

## What is implemented

- synthetic retail catalog generation with text, image-like, and structured attribute signals;
- a context encoder that mimics multi-modal product representation;
- separate heads for line prediction and ladder prediction, corresponding to the paper's staged reasoning;
- training and evaluation scripts with precision / recall / F1 for both structures.

## What is simplified

- the original paper's multi-agent prompting and enterprise routing are approximated with trainable neural heads over synthetic multi-modal features;
- real LLM extraction, catalog taxonomies, and production feedback loops are omitted;
- image evidence is represented by low-dimensional vectors rather than raw pixels.

## Run

```bash
python train.py
python test.py --checkpoint lines_ladders_toy.pt
```
