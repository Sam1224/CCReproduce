# QueryAgent-R1: E-Commerce Query Recommendation

Reproduction of **"QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation"** (arXiv 2606.05671).

## Quick Start

```bash
pip install -r requirements.txt
python train.py --epochs 20 --lambda_rl 0.5
python test.py --ckpt queryagent_r1_best.pt
```

## Architecture

```
User Query History (seq)
         │
   QueryEncoder (SASRec-style Transformer)
         │
   MemoryModule (attention over memory bank)
         │
   user_pref  ──→  GenerationHead ──→ query logits
                         │
                   sample query q'  ──→  Retriever ──→ retrieved_products
                                                              │
                                                    CVR Reward (Eq. 2)
                                                              │
                              REINFORCE: -R * log P(q' | seq)  (Eq. 3)
```

**Losses**: `L_total = L_CTR (cross-entropy) + λ * L_RL (REINFORCE)`

## Key Differences from Full System
- Simplified retriever (random sampling; real: ANN product retrieval engine)
- Toy synthetic data (real: Alibaba International query logs + product catalog)
- REINFORCE baseline (real: GRPO or PPO with larger policy model)

## Paper Results (Alibaba International)
- Query CTR: **+2.9%**
- Guided CVR: **+3.1%**
