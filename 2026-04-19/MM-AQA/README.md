# MM-AQA (Toy Reproduction)

Paper: **Knowing When Not to Answer: Evaluating Abstention in Multimodal Reasoning Systems** (arXiv:2604.14799)

This is a lightweight, runnable scaffold inspired by the paper’s benchmark idea:

- generate paired **answerable** vs **unanswerable** samples
- evaluate models with **abstention-aware** metrics

Because reproducing MMMU / MMLongBench-Doc faithfully requires large datasets and multimodal models, this code uses a synthetic “multimodal context” string but keeps the benchmark and metrics logic executable.

## Run

```bash
python eval.py
```
