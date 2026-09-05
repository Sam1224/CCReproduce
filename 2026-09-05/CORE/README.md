# CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation

Toy PyTorch reproduction for:

- **CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation**
- arXiv: https://arxiv.org/abs/2609.04083

The full paper trains large MLLM embedding/reranking models with synthesized compositional image-text candidate lists. This reproduction keeps the same core learning structure while replacing web-scale image generation and 2B/8B backbones with a compact runnable synthetic pipeline.

## What is implemented

- **Five-level compositional candidate lists**: L5 full match, L4 partial presence, L3 attribute error, L2 object error, L1 total mismatch.
- **Dual-encoder embedding model**: learns query and candidate representations and scores candidates with cosine similarity.
- **Cross-attentive reranker teacher**: a small Transformer reranker produces graded teacher distributions for each candidate list.
- **Rank-KL distillation**: listwise KL loss transfers the reranker ranking distribution into the embedding model.
- **Contrastive baseline**: optional CE-only training for comparison.
- **Toy train/test pipeline**: synthetic product/content examples, training loop, checkpointing, and ranking metrics.

## Quickstart

From this folder:

```bash
pip install -r requirements.txt
python train.py --epochs 3 --objective rank_kl
python test.py --checkpoint outputs/core_embed.pt
```

Expected output includes Recall@1, NDCG@5, and pairwise order accuracy over the five compositional levels.

## Mapping to the paper

| Paper component | This reproduction |
| --- | --- |
| MLLM embedding backbone | `COREEmbeddingModel`, a compact dual encoder |
| Cross-attentive reranker | `COREReranker`, a Transformer over query-candidate tokens |
| Five-level synthetic candidates | `CompositionalListDataset` |
| Rank-KL objective | `rank_kl_loss` in `model.py` |
| Compositional retrieval evaluation | `evaluate_rankings` in `test.py` |

## Notes and limitations

- This is a faithful implementation of the training logic, not a full 8B-scale reproduction.
- Real image generation, Qwen3VL backbones, LoRA fine-tuning, and the original benchmark loaders are represented by clean interfaces and synthetic features.
- To replace the toy data with production/e-commerce data, implement a dataset that returns the same `CompositionalBatch` fields used by `train.py`.
