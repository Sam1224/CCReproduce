# OneBar (Toy Reproduction)

Toy-but-runnable reproduction for:

- **OneBar: An End-to-End Content-Grounded Generative Query Recommendation Framework for E-Commerce Video Feeds** (arXiv:2606.15330)

## Goal

Reproduce the paper's *system logic* in a minimal setting:

- Build an **evidence schema** from (video content + product meta + user history).
- Apply a simple **prompt compression** step to fit an online-like context budget.
- Train an encoder–decoder **query generator** (BART-like in spirit; here: a small Transformer seq2seq).
- Add a lightweight **preference internalization** stage (PIOPD-like): on-policy generation + reward-weighted self-distillation to reduce reliance on an external reward model.

This is *toy but runnable*: it does not reproduce Kuaishou-scale infra, multimodal encoders, or real A/B traffic. It keeps the interfaces consistent (data/model/train/eval) and demonstrates the end-to-end generative pipeline.

## Quickstart

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## What you should see

- Supervised training improves exact-match accuracy.
- Preference-internalization stage further improves consistency under noisy history inputs.

## Files

- `data.py`: synthetic video-feed sessions (video/meta/history) -> target query.
- `model.py`: tiny Transformer seq2seq generator.
- `train.py`: supervised + preference-internalization training.
- `eval.py`: greedy decoding + exact-match / token-F1.
- `run_pipeline.py`: orchestration entry.
