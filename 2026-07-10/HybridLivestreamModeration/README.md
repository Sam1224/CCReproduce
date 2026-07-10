# HybridLivestreamModeration

Toy but runnable PyTorch reproduction for **Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching**.

## What is reproduced

This implementation keeps the paper's main system shape:

- a **preset violation detection** branch for known violations;
- a **reference matching** branch for novel / subtle violations;
- **teacher-to-student distillation** from synthetic teacher logits / hidden states;
- a compact **multimodal reranker** over retrieved candidates;
- a runnable **train / test** pipeline with a toy livestream dataset.

## Simplifications

- The production paper uses large-scale in-house livestream data, LLaVA-One-Vision, Swin, Whisper, MoCo+CLIP pretraining, and billion-scale HNSW indices.
- This reproduction replaces them with deterministic synthetic multimodal clip features and lightweight MLP encoders so the training loop remains fully runnable on CPU.
- The retrieval stage still mirrors the paper's design intent: query clips are matched against a reference policy memory and refined by a reranker.

## Files

- `dataset.py`: synthetic livestream clips plus a reference-bank builder
- `model.py`: hybrid classifier + retrieval + reranker + distillation loss
- `train.py`: trains the toy student model and saves checkpoint
- `test.py`: reports toy macro-F1, recall@precision, and retrieval recall

## Run

```bash
python train.py --cpu
python test.py --cpu
```
