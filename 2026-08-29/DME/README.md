# DME: Douyin Multimodal Embedding

PyTorch reproduction scaffold for the DME technical report on industrial multimodal video retrieval.

## Implemented modules

- Query text tower for user/search intent encoding.
- Video text tower for title/caption metadata.
- Frame tower with temporal attention over toy frame features.
- Fusion projection into a normalized retrieval embedding space.
- Binary retrieval training and evaluation on a toy commerce-video dataset.

## Quick start

```bash
python train.py --epochs 30 --output-dir runs/dme_toy
python test.py --checkpoint runs/dme_toy/checkpoint.pt
```

The production paper uses industrial-scale video data, large backbone encoders, and online retrieval infrastructure. This local version keeps the core interface and embedding-learning objective while replacing private video features with synthetic frame vectors.
