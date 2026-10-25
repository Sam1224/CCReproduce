# CaptionFuse Reproduction

This folder contains a lightweight PyTorch reproduction of the core idea in:

**CaptionFuse: Training-Free Composed Product Retrieval Via VLM Intent Captioning and Weighted Embedding Fusion**

The paper proposes **language-space fusion** for e-commerce composed image retrieval:

1. Read a reference product image and a sparse text modification.
2. Use a VLM to generate an attribute-rich product-title caption that states both what changes and what should be preserved.
3. Encode the reference image and generated caption independently.
4. Retrieve nearest neighbors with a weighted embedding fusion:

```text
q = normalize(alpha * image_embedding + (1 - alpha) * caption_embedding)
```

This implementation includes a fully runnable toy pipeline and backend interfaces that can be replaced by real VLM and embedding services.

## Files

- `dataset.py` — synthetic product catalog and composed-retrieval query generation.
- `captioner.py` — CaptionFuse VLM captioning interface plus a deterministic rule-based toy captioner.
- `model.py` — text/image encoders, weighted fusion, retrieval metrics.
- `train.py` — trains a compact CLIP-like toy encoder on synthetic product attributes.
- `test.py` — evaluates raw text vs CaptionFuse captions on composed retrieval.
- `requirements.txt` — minimal runtime dependencies.

## Quick start

```bash
pip install -r requirements.txt
python train.py --output checkpoints/toy_encoder.pt
python test.py --checkpoint checkpoints/toy_encoder.pt --alpha 0.15
```

Expected behavior on the toy benchmark: CaptionFuse captions should beat raw sparse modifications because they preserve unmentioned attributes from the reference product.

## Mapping to the paper

| Paper component | This reproduction |
| --- | --- |
| VLM intent captioning | `Captioner.generate()` / `RuleBasedCaptioner` |
| Product-title prompt | `captioner.py` creates concise title-like captions |
| Swappable embedding backbone | `ProductEncoder` interface in `model.py` |
| Weighted embedding fusion | `captionfuse_query()` in `model.py` |
| Retrieval evaluation | Recall@K and mAP in `model.py` |
| Cost-saving routing | `should_route_to_captioner()` in `captioner.py` |

The toy implementation does not claim to reproduce the paper's proprietary Nova/Titan backends or full FashionIQ/CIRCO/CIRR data preprocessing. Those parts are intentionally isolated behind interfaces so users can plug in real image/text encoders and VLM captioners.
