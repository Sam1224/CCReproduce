# QueryAgent-R1 (toy reproduction)

- Paper: **QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation** (arXiv:2606.05671, Alibaba International Digital Commercial Group)
- Core idea:
  - Close the **generation–retrieval gap**: a query that looks relevant (high CTR) may retrieve products that do not match purchase intent (low CVR).
  - Make query recommendation **retrieval-grounded**: generate a query, run real retrieval, and optimize the query for downstream item engagement.
  - Use an **agentic RL objective** (consistency reward) to jointly optimize query relevance and retrieval outcomes.

This folder implements a **toy but runnable** end-to-end pipeline that mirrors the *interfaces and key logic*:

- Synthetic e-commerce catalog + user behavior sequences.
- A small query policy that outputs a short query (2 tokens).
- A simple dense retriever over the catalog.
- RL fine-tuning with a consistency-style reward using offline signals.

> Notes
>
>- The original paper uses large proprietary logs, real inventory, and a much richer agent loop.
>- This toy reproduction keeps the key concept: **retrieval-grounded query generation + end-to-end reward**.

## Quickstart

```bash
python -m pip install -r requirements.txt

# Train (supervised warmup + RL fine-tuning)
python train.py --sup-epochs 3 --rl-epochs 3 --out runs/queryagent_r1.pt

# Evaluate (Q_EM / I_Hit@1 / Cons@1)
python evaluate.py --ckpt runs/queryagent_r1.pt
```

## Metrics (toy)

- **Q_EM**: exact match of the generated query tokens.
- **I_Hit@1**: whether the top-1 retrieved item is in the user’s synthetic purchase-intent set.
- **Cons@1**: `Q_EM * I_Hit@1`.

## Files

- `data.py`: synthetic catalog + user behavior generation.
- `model.py`: token vocab, memory abstraction, query policy, and dense retrieval.
- `train.py`: supervised warmup + REINFORCE-style RL fine-tuning.
- `evaluate.py`: offline evaluation metrics.
