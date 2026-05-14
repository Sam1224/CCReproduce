# CaseDrivenMASearch — Reproduction

**Paper:** A Case-Driven Multi-Agent Framework for E-Commerce Search Relevance  
**arXiv:** https://arxiv.org/abs/2605.05991  
**Authors:** Global E-Commerce Search Relevance Team, ByteDance

---

## Overview

This reproduction implements the three-agent closed-loop framework:

- **User Agent** — discovers bad cases via multi-turn conversational search simulation
- **Annotator Agent** — labels query-product relevance via multi-path chain-of-thought reasoning
- **Optimizer Agent** — root-cause analysis and automated resolution (policy / data / model gaps)

The toy data interface is aligned with the paper's described data format (query, product, label).

## Files

```
CaseDrivenMASearch/
├── README.md
├── requirements.txt
├── data/
│   └── toy_ecom_pairs.jsonl       # Toy query-product pairs
├── agents/
│   ├── user_agent.py              # User Agent: bad-case discovery
│   ├── annotator_agent.py         # Annotator Agent: multi-path relevance labeling
│   └── optimizer_agent.py         # Optimizer Agent: root-cause + resolution
├── models/
│   └── generative_relevance_model.py  # GRM: generative relevance model
├── pipeline.py                    # Full closed-loop pipeline
├── evaluate.py                    # Evaluation: precision / cost metrics
└── train.py                       # GRM training script
```

## Quick Start

```bash
pip install -r requirements.txt

# Run full closed-loop pipeline
python pipeline.py --data data/toy_ecom_pairs.jsonl --rounds 3

# Evaluate annotation quality
python evaluate.py --pred_file outputs/annotations.jsonl --gold_file data/toy_ecom_pairs.jsonl
```

## Architecture

```
Query + Product Pair
        │
        ▼
┌──────────────────┐
│   User Agent     │  Simulates user searches, identifies bad cases
│  (Bad-case Find) │  via multi-turn dialogue simulation
└────────┬─────────┘
         │ bad_cases
         ▼
┌──────────────────┐
│ Annotator Agent  │  Multi-path Chain-of-Thought reasoning
│ (GRM Training)   │  → generates relevance labels + rationale
└────────┬─────────┘
         │ labeled_data
         ▼
┌──────────────────┐
│ Optimizer Agent  │  Root-cause analysis:
│ (Resolution)     │  Policy gap / Data gap / Model gap
└────────┬─────────┘
         │ fixes (policy update / synthetic data / hyper-tuning)
         ▼
┌──────────────────┐
│      GRM         │  Generative Relevance Model (fine-tuned LLM)
│  (Fine-tuning)   │  Predicts relevance label + explanation
└──────────────────┘
```
