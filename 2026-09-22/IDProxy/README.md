# IDProxy reproduction

Implements the central IDProxy idea: a multimodal proxy embedding is generated from item content, aligned into the item-ID embedding space with a coarse-to-fine module, and consumed by a CTR ranker when items are in cold start.

Run:

```bash
python train.py --epochs 3
python test.py
```
