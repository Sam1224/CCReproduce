# QueryAgent-R1: E-Commerce Query Recommendation

Toy reproduction of **"QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation"** (arXiv:2606.05671, Alibaba International).

## Architecture

```
User history ──► [Memory-Augmented Agent]
                        │
              ┌─────────▼──────────┐
              │  Generate candidate│
              │  query (LLM/policy)│
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  Chain-of-Retrieval│◄── Inventory API
              │  (validate query)  │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  Consistency Reward│
              │  (RL training)     │
              └─────────┬──────────┘
                        │
              Refined query recommendation
```

## Files

- `agent.py` — memory-augmented query generation agent
- `retriever.py` — toy product retrieval API (chain-of-retrieval)
- `reward.py` — consistency reward computation (query relevance + CVR alignment)
- `rl_trainer.py` — RL training loop (REINFORCE / PPO-lite)
- `data.py` — toy dataset (user history + product inventory)
- `evaluate.py` — offline evaluation (Q_EM, I_Hit@K, Cons@K)
- `requirements.txt` — dependencies

## Quick Start

```bash
pip install -r requirements.txt
# Generate toy dataset
python data.py
# Train agent with RL
python rl_trainer.py --epochs 5 --output checkpoints/
# Evaluate
python evaluate.py --checkpoint checkpoints/best.pt
```

## Paper Reference

arXiv: https://arxiv.org/abs/2606.05671

**Key metrics (paper):**
- Industrial: Cons@1 0.025 → 0.117 (+368%)
- Production A/B: Query CTR +2.9%, Guided CVR +3.1%
