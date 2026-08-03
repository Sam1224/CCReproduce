# GALA (toy reproduction)

This folder provides a runnable PyTorch toy reproduction for **GALA: Generative Aligned Learning for Adaptive Multimodal Representation in the Taobao Shangou Recommender System**.

## What is preserved
- query-image-text triplet alignment pretraining
- reward-weighted alignment stage that mimics GRPO-style behavioral optimization
- adaptive gate that fuses ID and multimodal item representations for ranking

## What is simplified
- synthetic instant-commerce catalog and user behavior data
- lightweight MLP/GRU encoders instead of the production-scale retrieval and serving stack
- reward alignment implemented as a stable weighted policy objective rather than a full RL system

## Run
```bash
python train.py --epochs_stage1 3 --epochs_stage2 2 --epochs_stage3 4
python test.py --checkpoint gala_toy.pt
```
