# QueryAgent-R1: PyTorch Reproduction

Faithful reproduction of "QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation" (arXiv:2606.05671, Alibaba International Digital Commercial Group).

## Architecture Overview

```
User History (long-term) ──► Interest Graph Memory ──► User Embedding
                                                              │
                                                              ▼
                                              ┌──────────────────────────┐
                                              │  LLM-based Query Generator│
                                              │  (+ Retrieved Context)   │
                                              └──────────────────────────┘
                                                              │ candidate queries
                                                              ▼
                                              ┌──────────────────────────┐
                                              │   Product Retrieval API  │
                                              │   (BM25 / Dense)         │
                                              └──────────────────────────┘
                                                              │ retrieved products
                                                              ▼
                                              ┌──────────────────────────┐
                                              │  Consistency Reward      │
                                              │  = α·CTR_proxy           │
                                              │  + β·CVR_proxy           │
                                              └──────────────────────────┘
                                                              │
                                                    RL Fine-tuning (GRPO)
```

## Files

| File | Description |
|------|-------------|
| `model.py` | Interest Graph memory module + LLM-based query generator |
| `retrieval.py` | Product retrieval interface (BM25 toy version) |
| `reward.py` | Consistency reward computation |
| `data.py` | Toy e-commerce dataset (query–user–product triples) |
| `train.py` | GRPO-based RL training loop |
| `eval.py` | Offline evaluation (CTR proxy, CVR proxy, MRR) |
| `run_demo.sh` | End-to-end demo script |

## Quick Start

```bash
pip install -r requirements.txt
python train.py --config configs/toy.yaml
python eval.py --checkpoint checkpoints/best.pt
```

## Paper Reference

> Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng.
> "QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation."
> arXiv:2606.05671, 2026.
