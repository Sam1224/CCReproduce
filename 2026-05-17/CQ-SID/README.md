# CQ-SID + EG-GRPO (Toy Reproduction)

This folder contains a **toy, runnable** reproduction of the main ideas in:

- **Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL** (arXiv:2605.14434)

What is implemented:
- A minimal **Residual Quantized VAE (RQ-VAE)** for assigning cluster-like discrete codes.
- A tiny **Transformer seq2seq** model for **Query→SID** mapping (SFT).
- A simplified **EG-GRPO-like** group-relative policy optimization loop with **expert (ground-truth) injection**.

Notes:
- This is intentionally small-scale and uses a synthetic corpus to keep training fast.
- It is designed so the data/model/training interfaces match the paper’s pipeline structure.

## Quickstart

```bash
pip install -r requirements.txt  # or install torch + numpy manually
python3 train_rqvae.py
python3 train_sft.py --stage query2sid
python3 train_sft.py --stage user_query2sid --out outputs/sft_user.pt
python3 train_eg_grpo.py --sft outputs/sft_user.pt
```
