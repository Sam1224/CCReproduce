# MA-Bench (toy reproduction)

Toy reproduction skeleton for:

- **MA-Bench: Towards Fine-grained Micro-Action Understanding** (arXiv:2603.26586)

## Idea captured

We simulate micro-action understanding as fine-grained temporal classification:

- A sequence of micro-action tokens is the "video".
- The task is to classify the current micro-action under context.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-27/ma_bench
python3 -m pip install -r requirements.txt
python3 train.py --epochs 10
python3 test.py
```
