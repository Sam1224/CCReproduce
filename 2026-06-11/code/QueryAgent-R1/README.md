# QueryAgent-R1: E-Commerce Query Recommendation

Reproduction of: [QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation](https://arxiv.org/abs/2606.05671)  
Alibaba International Digital Commerce Group, June 4, 2026

## Structure

```
QueryAgent-R1/
├── model.py        # Agent model: LLM + memory abstraction + retrieval interface
├── retrieval.py    # Product retrieval module (BM25 + dense)
├── rl_train.py     # GRPO-based RL training with consistency reward
├── data.py         # Dataset: user history + product catalog
├── eval.py         # Evaluation: CTR, CVR, query relevance
└── README.md
```

## Setup

```bash
pip install torch transformers accelerate faiss-cpu rank-bm25
```

## Quick Start

```bash
# Train with GRPO RL
python rl_train.py --data data/ --epochs 10 --toy

# Evaluate query quality
python eval.py --data data/ --model checkpoints/queryagent_best.pt --toy
```
