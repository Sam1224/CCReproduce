# Symbal Reproduction

This folder implements a practical PyTorch reproduction of **Symbal: Detecting Systematic Misalignments in Model-Generated Captions**.

The original paper link points to `https://github.com/Stanford-AIMI/Symbal`, but the repository returned 404 during verification on 2026-07-17, so this implementation recreates the core pipeline from the paper.

## What is implemented

The code follows the two-stage Symbal design. Stage 1 splits captions into sentence-level textual facts, embeds and clusters them with spherical k-means, scores clusters by low image-text alignment, and summarizes the top cluster as the recurring textual error. Stage 2 takes records containing that textual error, clusters their image embeddings, ranks image clusters by misalignment, and summarizes the associated visual feature.

The production paper uses large off-the-shelf encoders and MLLM/LLM scorers such as Qwen, OpenCLIP, XRayCLIP, MedSigLIP, and MedGemma. This reproduction keeps the same interfaces but uses lightweight hash text embeddings, trainable PyTorch image projections, and toy metadata summarizers so the full pipeline runs locally.

## Quick start

```bash
pip install -r requirements.txt
python toy_dataset.py --output toy_symbal.json
python train.py --data toy_symbal.json --checkpoint symbal_projection.pt
python test.py --data toy_symbal.json --checkpoint symbal_projection.pt --top-k 3
```

Expected output contains a predicted pair similar to `textual_error=white tablecloth` and `visual_feature=table`.

## Files

`toy_dataset.py` creates a dataset with image tags, numeric image features, and injected systematic caption errors. `symbal.py` contains the reusable pipeline. `train.py` trains a small image projection with contrastive alignment. `test.py` runs end-to-end detection.
