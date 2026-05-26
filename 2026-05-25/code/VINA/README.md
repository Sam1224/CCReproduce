# VINA — PyTorch Reproduction

**Paper:** "Video as Natural Augmentation: Towards Unified AI-Generated Image and Video Detection"  
**arXiv:** https://arxiv.org/abs/2605.21977  
**Score:** 83/100

This folder contains a faithful toy reproduction of the VINA framework in PyTorch.

## Architecture

VINA jointly trains an AIGC detector on both still images and video frames, using:

1. **Cross-Modal Data Mixing:** Video frames (with codec/compression artifacts) serve as natural augmentations for the image detector
2. **Cross-Modal Supervised Contrastive Loss:** Aligns representations of images and video frames from the same source (real/fake) in a unified embedding space
3. **Unified Inference:** Single model classifies both images and video frames without separate heads

## Files

| File | Description |
|------|-------------|
| `dataset.py` | Toy image+video dataset with synthetic real/fake generation |
| `model.py` | ResNet-based VINA detector with contrastive projection head |
| `train.py` | Joint training with CE + Cross-Modal Contrastive loss |
| `eval.py` | Evaluation on image and video frame benchmarks |
| `requirements.txt` | Python dependencies |

## Quickstart

```bash
cd CCReproduce/2026-05-25/code/VINA
pip install -r requirements.txt
python train.py --epochs 10 --batch_size 32
python eval.py --ckpt checkpoints/vina_best.pt
```

## Toy Dataset

The toy dataset simulates the real/fake distribution:
- **Real images:** Natural image patches with smooth frequency spectra
- **Fake images:** Synthesized with GAN-like high-frequency artifacts + spectral peaks at specific bands
- **Real video frames:** Real images with simulated codec compression (JPEG quality variation, blur)
- **Fake video frames:** Fake images with additional video-specific processing

## Training Loss

The total loss combines cross-entropy and cross-modal contrastive:

```
L_VINA = L_CE + λ * L_SCL

L_SCL = Supervised Contrastive Loss across image-video pairs from same source
```

## Notes

- This is a **toy reproduction** with synthetic data. Replace `ToyAIGCDataset` with real data loaders for actual evaluation.
- The ResNet backbone can be swapped for ViT/DINOv2 as in the paper.
- Contrastive loss parameters (temperature τ, weight λ) were set to paper defaults where reported.
