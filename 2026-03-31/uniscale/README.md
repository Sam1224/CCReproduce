# UniScale (reproduction-oriented implementation)

Reference paper: **UniScale: Synergistic Entire Space Data and Model Scaling for Search Ranking** (arXiv:2603.24226)

This folder implements a *runnable* UniScale-style pipeline with:

- **ES^3 (Entire-Space Sample System)**: a data constructor that expands exposed search samples with (a) unexposed candidates, (b) cross-domain supervision via hierarchical label attribution, and (c) cross-domain “searchification” (feature alignment into a search-style schema).
- **HHSFT (Heterogeneous Hierarchical Sample Fusion Transformer)**: a heterogeneous-token Transformer that models (a) hierarchical feature interaction and (b) entire-space user-interest fusion via a domain-routed expert layer + domain-aware gated attention.

> Note: the paper describes production optimizations (feature pre-hashing, attention kernel fusion, RDMA comms, fp16 quantization). These are **out of scope** for a pure PyTorch reproduction and are provided as **marked pseudocode** at the end of this README.

## What is implemented (non-toy)

- Heterogeneous feature tokenization into **typed tokens** (user / request-context / item / domain / cross-domain interest).
- Hierarchical interaction stack: **intra-group attention → cross-group attention**.
- Domain-routed expert fusion (hard/soft routing) for cross-domain interest.
- ES^3 data builder that produces request-level candidate sets and constructs:
  - unexposed negatives
  - delayed conversion attribution → denser supervision
  - cross-domain behavior mapping into aligned features
- Training loop with:
  - **pointwise BCE** for click/purchase probability
  - **pairwise ranking loss** inside each request
  - evaluation: AUC and NDCG@K

## Quickstart

```bash
cd ccreproduce_repo/2026-03-31/uniscale
python3 -m pip install -r requirements.txt
python3 train.py --epochs 5 --requests 4000 --candidates 32
python3 test.py --ckpt checkpoints/uniscale.pt
```

## File layout

- `dataset.py`: ES^3-style synthetic data + real-data hooks
- `model.py`: HHSFT-style model
- `train.py`: training + eval
- `test.py`: load checkpoint and run eval

## Pseudocode (explicitly NOT implemented)

### Production feature hashing / pre-processing

```text
# NOT IMPLEMENTED
# Build offline feature dictionary / hashing table
for feature in sparse_features:
  hash_id = murmurhash(feature)
  store(hash_id)

# Online serving:
#   convert raw feature -> hash_id -> embedding lookup
```

### Kernel fusion / RDMA / fp16 quantization

```text
# NOT IMPLEMENTED
# 1) Fuse attention projections + softmax into one kernel
# 2) Use RDMA to fetch remote embeddings / MoE expert params
# 3) Quantize activations/weights to fp16 with calibration
```
