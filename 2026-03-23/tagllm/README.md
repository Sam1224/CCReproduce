# TagLLM (toy reproduction)

Toy reproduction skeleton for:

- **TagLLM: A Fine-Grained Tag Generation Approach for Note Recommendation** (arXiv:2603.21481)

## Idea captured

We model “tag generation for note recommendation” as **multi-label tag prediction** conditioned on note text.

This toy version captures:

- Fine-grained tags (multi-label)
- Text → tag logits via a lightweight encoder
- Micro-F1 evaluation (typical for tagging)

## Not covered

- True LLM prompting / instruction tuning
- Retrieval-augmented tagging
- Large-scale user behavior modeling for recommendation

## Quickstart

```bash
cd ccreproduce_repo/2026-03-23/tagllm
python3 -m pip install -r requirements.txt
python3 train.py --epochs 8
python3 test.py
```
