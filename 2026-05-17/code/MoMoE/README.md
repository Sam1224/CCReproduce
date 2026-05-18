# MoMoE — Code Reproduction

Reproduction of: **MoMoE: Mixture of Moderation Experts Framework for AI-Assisted Online Governance**  
ArXiv: https://arxiv.org/abs/2505.14483  
Venue: EMNLP 2025  
Official code: https://github.com/scuba-illinois/MoMoE

## Structure

```
MoMoE/
├── README.md              # This file
├── requirements.txt       # Dependencies
├── data/
│   └── dataset.py         # Dataset loading (Reddit/custom)
├── models/
│   ├── allocator.py       # Allocate operator (RoBERTa-base)
│   ├── expert.py          # Predict operator (SLM experts)
│   └── explainer.py       # Explain operator (LLM-based)
├── operators/
│   └── aggregate.py       # Aggregate operator (weighted / majority vote)
├── momoe.py               # Full MoMoE pipeline
├── train_allocator.py     # Train the Allocate operator
├── train_experts.py       # Fine-tune community/norm-violation experts
└── evaluate.py            # Evaluation script
```

## Quick Start

```bash
pip install -r requirements.txt

# Train allocator (community-based)
python train_allocator.py --mode community --data_path data/train.jsonl

# Train experts
python train_experts.py --allocator_type community --n_experts 7

# Run full pipeline evaluation
python evaluate.py --mode community --data data/test.jsonl
```

## Pipeline Overview

```
Input Post
    ↓
[Allocate] RoBERTa-base → expert weights w_1...w_K
    ↓
[Predict]  K specialized SLM experts → predictions p_1...p_K
    ↓
[Aggregate] weighted dot product OR majority vote → final prediction
    ↓
[Explain]  GPT-4o / local LLM → 3-level JSON explanation
    ↓
Output: {label, confidence, explanation}
```
