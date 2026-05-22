# Airbnb Cold-Start Synthetic Data for NL Search (Toy Reproduction)

This folder is a lightweight, **PyTorch**-based reproduction scaffold for the ideas in:

**"Bridging the Cold-Start Gap: LLM-Powered Synthetic Data Generation for Natural Language Search at Airbnb"** (arXiv:2605.21812)

It implements a minimal end-to-end pipeline with:

- **Toy marketplace data** (listings + booking-session-derived contrastive pairs)
- **Seed-guided contrastive query generation** (rule-based stub, with clear LLM hook points)
- **Two labeling strategies**
  - *Contrastive-by-construction* labels (pos/neg pair)
  - *Virtual Judge* heuristic labels (stubbed scoring)
- **Embedding-based retrieval model** (dual encoder with `nn.EmbeddingBag` + pairwise ranking loss)

## Files

- `toy_data/seed_queries.txt`: seed queries (user research)
- `toy_data/listings.jsonl`: toy listing catalog
- `toy_data/sessions.jsonl`: toy sessions with contrastive listing pairs
- `generate_data.py`: generate synthetic (query, pos, neg) training triples
- `dataset.py`: dataset + vocab + utilities
- `model.py`: simple dual-encoder retrieval model
- `train.py`: train retrieval model
- `test.py`: evaluate pairwise accuracy

## Quickstart

```bash
cd CCReproduce/2026-05-22/airbnb_synth_search
python3 generate_data.py --out toy_data/train_pairs.jsonl
python3 train.py --train toy_data/train_pairs.jsonl --epochs 5
python3 test.py --data toy_data/train_pairs.jsonl --ckpt checkpoints/model.pt
```

## Notes vs. the paper

- The paper uses LLM prompting for query/label generation. Here we provide a **deterministic stub** so the pipeline runs offline.
- The code is structured so swapping `RuleBasedLLMStub` with a real LLM client is localized to `dataset.py`.
