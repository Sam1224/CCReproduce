# CAPE: Co-occurrence-Aware Pool Expansion

PyTorch reproduction scaffold for **Generative Recommenders Forget the Long Tail: Restoring Memorization in Retrieval and Ranking**.

This implementation follows the paper's deployable plugin setting:

- keep a frozen / pretrained generative recommender that scores semantic item identifiers;
- run constrained beam search to obtain the backbone candidate pool;
- build an item-level sparse co-occurrence memory from training interactions;
- add tail items outside the beam by co-occurrence support;
- score added candidates by teacher forcing through the frozen generator;
- rank with `minmax(log_likelihood) + lambda * minmax(co_occurrence_evidence)` while keeping the generator's top-1 item.

The included dataset is synthetic but uses the same interfaces as the training and evaluation scripts, so the code can be swapped to Amazon / LastFM style sequential recommendation data by replacing `ToySequentialDataset`.

## Files

- `dataset.py`: toy sequential recommendation dataset and sparse co-occurrence memory.
- `model.py`: compact GRU-based generative recommender over semantic item IDs.
- `cape.py`: CAPE candidate expansion and calibrated ranking.
- `train.py`: trains the backbone recommender on the toy dataset.
- `evaluate.py`: compares backbone vs. CAPE on overall and tail HR@10 / NDCG@10.

## Quick start

```bash
python train.py --epochs 3 --output ./checkpoints/backbone.pt
python evaluate.py --checkpoint ./checkpoints/backbone.pt --lambda-mem 0.35
```

The script prints overall metrics and tail metrics. The toy numbers are not meant to reproduce the paper tables; they validate the full pipeline and the method logic.

