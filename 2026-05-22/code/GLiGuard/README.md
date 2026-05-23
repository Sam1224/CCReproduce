# GLiGuard — Usage Reference & E-Commerce Adaptation Guide

> Paper: [GLiGuard: Schema-Conditioned Classification for LLM Safeguard](https://arxiv.org/abs/2605.07982)  
> Official code: https://github.com/fastino-ai/GLiGuard  
> Official model: https://huggingface.co/fastino/gliguard-LLMGuardrails-300M  
> License: Apache 2.0

The official repository contains a **complete, production-ready implementation**. This folder provides:
1. An e-commerce content governance adaptation example (`ecom_guard.py`)
2. A batch inference script for influencer (达人) content moderation (`batch_moderate.py`)
3. Evaluation harness against a toy dataset (`eval_ecom.py`)

## Installation

```bash
pip install gliner2 transformers torch
```

## Quickstart (official interface)

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("fastino/gliguard-LLMGuardrails-300M")

# Single call returns all safety aspects simultaneously
result = model.classify_text(
    text="Buy this slimming tea — lose 10kg in 7 days, guaranteed!",
    task_names=["prompt_safety", "harm_categories", "jailbreak"],
    candidate_labels=["safe", "unsafe", "health_claim", "misleading_ad"]
)
print(result)
```
