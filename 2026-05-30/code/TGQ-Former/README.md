# TGQ-Former — Code Reproduction

Paper: "Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation"  
arXiv: 2605.17366

## Key Idea

Product images on e-commerce platforms contain promotional overlays and background clutter that degrade retrieval quality. TGQ-Former **adaptively disentangles** product-relevant visual features from noise by producing two complementary query streams:

- **Metadata-Anchored Queries:** Strongly guided by product metadata; capture metadata-consistent evidence.
- **Exploratory Queries:** Weakly guided; capture complementary visual patterns beyond the metadata.

**Dual-Gated Vector Modulation (DGVM)** calibrates both streams using cross-modal agreement and visual saliency. **Redundancy-Reduction Regularizer (R³)** enforces complementarity.

## Structure

```
TGQ-Former/
├── model.py   # Full architecture: TGQFormerDualStream, DGVM, R³, TGQFormerRetrieval
├── train.py   # Training with contrastive loss + R³ penalty
└── README.md
```

## Setup

```bash
pip install torch transformers
```

## Quick Start

```bash
python train.py --epochs 20 --batch_size 64 --alpha_r3 0.1
```

## Architecture

```
Product Image → Visual Encoder (frozen ViT/CLIP) → Visual Tokens
Product Metadata → Text Encoder → Metadata Repr
                                       ↓
                         TGQ-Former Dual Stream
                         ├── Meta Q-Former (strong guidance)  → Q_meta
                         └── Explore Q-Former (weak guidance) → Q_explore
                                       ↓
                         Dual-Gated Vector Modulation (DGVM)
                         (calibrate via cross-modal agreement + saliency)
                                       ↓
                         Pool + Concat + Project → Item Embedding
                                       ↓
                         InfoNCE Contrastive Loss + R³ Penalty
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_r3` | 0.1 | Weight of redundancy-reduction regularizer |
| `num_meta_queries` | 16 | Number of metadata-anchored query tokens |
| `num_explore_queries` | 16 | Number of exploratory query tokens |
| `num_layers` | 6 | Number of TGQ-Former layers |
| `query_dim` | 256 | Query representation dimension |
| Temperature | 0.07 | InfoNCE contrastive temperature |
