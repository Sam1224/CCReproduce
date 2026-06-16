# AIR: Atomic Intent Reasoning

Toy reproduction of **"Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations"** (arXiv:2606.10357, Kuaishou Technology).

## Architecture

```
Content behaviors  ──► [LLM Offline]  ──► Intent Atoms Index
                                              │
                              ┌───────────────┘
User request ──────────────►  │  Online: retrieve & compose atoms
                              └──────► Cross-domain item ranking
```

## Files

- `intent_atomizer.py` — offline LLM-based intent decomposition
- `atom_index.py` — FAISS-based intent atom indexer
- `online_composer.py` — online retrieval + composition for item ranking
- `data.py` — toy dataset (content events + e-commerce items)
- `train_ranker.py` — ranker fine-tuning with composed intent vectors
- `evaluate.py` — NDCG / Hit Rate evaluation
- `requirements.txt` — dependencies

## Quick Start

```bash
pip install -r requirements.txt
# Step 1: Offline — atomize intent from content behaviors
python intent_atomizer.py --behaviors data/content_behaviors.json --output data/atoms.json
# Step 2: Build atom index
python atom_index.py --atoms data/atoms.json --index data/atom_index.faiss
# Step 3: Online inference (simulate)
python online_composer.py --index data/atom_index.faiss --items data/ecom_items.json
# Step 4: Evaluate
python evaluate.py --predictions data/predictions.json --ground_truth data/ground_truth.json
```

## Paper Reference

arXiv: https://arxiv.org/abs/2606.10357
