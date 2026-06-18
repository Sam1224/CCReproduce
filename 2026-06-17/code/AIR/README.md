# AIR: Atomic Intent Reasoning

Reproduction of **"Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations"** (arXiv 2606.10357).

## Quick Start

```bash
pip install -r requirements.txt
python train.py --epochs 20 --lr 1e-3
python test.py --ckpt air_best.pt
```

## Architecture

```
Content Domain (Video Watch)        Commerce Domain (Purchase)
        │                                    │
  SessionEncoder                      SessionEncoder
        │                                    │
  session_emb ──→ OnlineIntentComposer      u_commerce
                        │  (ANN over atom prototypes)
                   u_intent  ──→  CrossDomainTransfer ──→ u_transferred
                                                │
                                         concat + MLP
                                                │
                                           u_final ──→ dot(v_item) ──→ score
```

**Loss**: `L_total = L_rank (BPR) + λ * L_sc (cosine consistency)`

## Key Differences from Full System
- Atom prototypes are randomly initialized (paper uses LLM-generated offline reasoning)
- Toy synthetic dataset (paper uses Kuaishou E-commerce logs)
- Simplified online composition (paper uses production FAISS cluster)
