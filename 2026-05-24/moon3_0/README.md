# MOON3.0 — Toy Reproduction (PyTorch)

This folder is a **minimal runnable scaffold** inspired by:

**"MOON3.0: Reasoning-aware Multimodal Representation Learning for E-commerce Product Understanding"** (arXiv:2604.00513)

The paper targets **e-commerce product representation learning** with three core ideas:

- **Multi-head modality fusion** to integrate raw image + title signals
- **Attribute deconstruction (reasoning)** to explicitly produce fine-grained attributes before embedding
- **Joint contrastive + RL training** to explore better reasoning strategies
- **Fine-grained residual enhancement** to preserve local details through deep networks

This toy repo focuses on the *architecture shape* and *training pipeline*. The RL part is left as a documented stub (see `train.py`).

## Files

- `dataset.py`: synthetic e-commerce-like products + query/pos/neg triplets
- `model.py`: a lightweight MOON3.0-style model
- `train.py`: trains with contrastive + attribute generation loss (RL stub)
- `test.py`: evaluates retrieval Recall@K on the toy catalog
- `requirements.txt`

## Quickstart

```bash
cd CCReproduce/2026-05-24/moon3_0
python3 -m pip install -r requirements.txt
python3 train.py --epochs 5
python3 test.py --ckpt checkpoints/moon3_0.pt
```

## Notes on faithfulness

- **Kept**: explicit attribute deconstruction head, multimodal fusion, contrastive retrieval objective, a residual enhancement block.
- **Simplified**: vision/text encoders are tiny; no large MLLM; RL (GRPO-like) optimization is provided as pseudocode.
- **How to extend**: replace `ToyProductCatalog` with real product images/titles and plug in a real VLM/MLLM encoder.
