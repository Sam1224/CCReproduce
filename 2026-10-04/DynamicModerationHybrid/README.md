# DynamicModerationHybrid

PyTorch reproduction scaffold for **Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching**.

This implementation follows the paper's dual-path production design:

- supervised multimodal classification for known policy categories;
- reference-based similarity matching for novel/subtle violations;
- MLLM-style teacher distillation into lightweight student encoders/classifiers;
- clip-level fusion over visual, audio, and ASR/text features;
- toy data generation with the same interfaces as the training and evaluation scripts.

The toy dataset is synthetic and intentionally small so the full pipeline can run on CPU. Replace `ToyLivestreamDataset` with production feature readers that emit the same keys: `visual`, `audio`, `text`, `label`, `reference_id`.

## Quick start

```bash
pip install -r requirements.txt
python train.py --epochs 3 --batch-size 32 --out-dir runs/demo
python evaluate.py --checkpoint runs/demo/best.pt
python infer.py --checkpoint runs/demo/best.pt
```

## Files

- `data.py`: synthetic livestream clip dataset and reference bank builder.
- `model.py`: multimodal student encoder, teacher stub, classifier, contrastive matcher, and fusion logic.
- `train.py`: end-to-end distillation + classification + similarity training.
- `evaluate.py`: precision/recall metrics at target precision, plus branch contribution.
- `infer.py`: example real-time clip inference API.

## Notes on faithful reproduction

The paper's production data, violation taxonomy, proprietary MLLM teacher, and online serving stack are not public. This reproduction preserves the algorithmic structure and training objectives, while using synthetic features and a teacher stub. Replace the teacher stub with an actual MLLM scoring service and the toy data with sampled clip features for a production-grade reproduction.
