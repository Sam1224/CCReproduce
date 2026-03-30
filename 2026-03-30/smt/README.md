# smt (toy reproduction)

This folder is a **toy, runnable reproduction** of the engineering concept in:

- **Design Once, Deploy at Scale: Template-Driven ML Development for Large Model Ecosystems** (arXiv:2603.24963)

The paper is largely about *systems and organization*: a **Standard Model Template (SMT)** that lets you ship changes across a large fleet of ranking models with shared building blocks.

## What this reproduction covers

- A minimal **template-driven ranking model** builder (`StandardModelTemplate`).
- Composable blocks: feature encoders, interaction layers, loss heads, optional calibration.
- Two example configs that build different models from the same template.

## Quickstart

```bash
cd 2026-03-30/smt
python3 -m pip install -r requirements.txt
python3 train.py
```

## Files

- `dataset.py`: toy ranking dataset with categorical + numeric features
- `blocks.py`: reusable building blocks
- `model.py`: the template that composes blocks
- `train.py`: trains two configs and compares
- `test.py`: smoke test

## Notes / limitations

- This is not meant to replicate Meta’s production system.
- The goal is to provide a clear reference for “template-driven” ML development patterns.
