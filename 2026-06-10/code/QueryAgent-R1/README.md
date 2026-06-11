# QueryAgent-R1: E-Commerce Query Recommendation via Chain-of-Retrieval RL

Reproduction of **QueryAgent-R1** (arXiv: 2606.05671)  
*Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation*  
Dike Sun, Zheng Zou, et al. — Alibaba International Digital Commercial Group

---

## Paper Summary

QueryAgent-R1 addresses a fundamental misalignment in e-commerce query recommendation:
- **Problem**: Existing methods optimize query-level relevance (click-through rate / CTR) but ignore whether retrieved products align with downstream user preferences, leading to high CTR but low conversion rate (CVR).
- **Solution**: A memory-augmented agentic RL framework that grounds query generation in real inventory retrieval, validates queries against retrieved products, and uses a **consistency reward** to jointly optimize query relevance and downstream engagement.
- **Key components**:
  1. Memory abstraction module → interest graph from long-term user history
  2. Chain-of-retrieval agent → generate query → retrieve products → validate/refine → repeat
  3. RL with consistency reward (joint CTR + CVR signal)

---

## Repository Structure

```
QueryAgent-R1/
├── README.md
├── requirements.txt
├── data.py            # Toy e-commerce dataset interface
├── model.py           # QueryAgent-R1 model (memory encoder + policy LLM + retriever)
├── train.py           # RL training with GRPO-style consistency reward
├── evaluate.py        # Offline evaluation: query quality, retrieval alignment
└── demo.py            # Interactive demo for query generation + retrieval loop
```

---

## Quick Start

```bash
pip install -r requirements.txt
python train.py --epochs 5 --batch_size 16
python evaluate.py --checkpoint checkpoints/best.pt
python demo.py --user_id 42
```
