# diet (toy reproduction)

This folder is a **lightweight, runnable reproduction skeleton** of the core idea in:

- **DIET: Learning to Distill Dataset Continually for Recommender Systems** (arXiv:2603.24958)

## What this reproduction covers

DIET’s key idea is to make recsys iteration cheaper by maintaining a **tiny distilled training set** that is **continually updated** as new interaction data arrives, so future model re-training doesn’t need to replay the full history.

The original paper uses a more sophisticated bilevel formulation and large-scale recsys setups. Here we implement a faithful *mechanism-level* skeleton on a toy implicit-feedback matrix factorization model:

- A stream of interaction samples is simulated.
- We keep a small **distilled subset** of samples (fixed indices) with **learnable per-sample weights**.
- A bilevel step updates those weights so that a 1-step inner update on distilled data better matches the loss on a full batch.

## Quickstart

```bash
cd 2026-03-30/diet
python3 -m pip install -r requirements.txt
python3 train.py
```

## Files

- `dataset.py`: toy implicit-feedback data + stream iterator
- `model.py`: tiny MF model (user/item embeddings)
- `distill.py`: 1-step bilevel “distilled weights” update
- `train.py`: end-to-end demo training loop
- `test.py`: smoke test (no guarantees on metrics)

## Notes / limitations

- This is **not** the full-scale DIET reproduction. It is intended as a clean reference implementation of the distilled-data + continual update loop.
- The distilled set here is implemented as a small subset with learnable weights; extending it to fully synthetic learnable examples is left as a TODO hook in `distill.py`.
