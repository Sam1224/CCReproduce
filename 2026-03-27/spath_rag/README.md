# S-Path-RAG: Semantic-Aware Shortest-Path RAG for Multi-Hop KGQA

This repository contains a PyTorch implementation skeleton for S-Path-RAG.

## Architecture
- **S-Path Retrieval**: Enumerates shortest paths in KG.
- **Path Scoring**: Semantic alignment between questions and paths.
- **Latent Injection**: Cross-attention mechanism to inject path latents into LLM.

## Usage
1. Prepare KG and question data.
2. Run `python train.py` to start training.
3. Run `python test.py` for inference.
