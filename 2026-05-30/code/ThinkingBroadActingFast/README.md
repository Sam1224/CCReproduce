# Thinking Broad, Acting Fast — Code Reproduction

Paper: "Thinking Broad, Acting Fast: Latent Reasoning Distillation from Multi-Perspective Chain-of-Thought for E-Commerce Relevance"  
arXiv: 2601.21611 | WWW 2026

## Structure

```
ThinkingBroadActingFast/
├── model.py      # Core architecture: Teacher LLM, Student model, Distillation loss
├── train.py      # Two-stage training pipeline (Stage 1: CoT gen, Stage 2: distillation)
├── eval.py       # Evaluation with NDCG@10, AUC, Accuracy
└── README.md
```

## Setup

```bash
pip install torch transformers scikit-learn
```

## Quick Start

```bash
# Stage 1 + 2 combined (toy data auto-generated if no data file exists)
python train.py --epochs 5 --batch_size 16 --lambda_distill 0.5

# Evaluate
python eval.py
```

## Data Format (JSONL)

```json
{
  "query": "running shoes for women",
  "product_title": "Nike Air Zoom Women's Running Shoes",
  "product_attrs": "color: blue, size: US 7",
  "label": 1,
  "teacher_reasoning": {
    "semantic_match": "...",
    "user_intent": "...",
    "attribute_match": "...",
    "long_tail_coverage": "..."
  }
}
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_distill` | 0.5 | Weight of latent alignment loss |
| `num_perspectives` | 4 | Number of CoT perspectives |
| `hidden_dim` | 768 | Encoder hidden dimension |
| `lr` | 2e-5 | Learning rate for student |

## Architecture Notes

- **Teacher LLM**: Any instruction-tuned LLM (Qwen2, LLaMA-3, GPT-4). Called offline only.
- **Student Model**: Lightweight BERT-based dual encoder with latent reasoning head.
- **Distillation**: Latent MSE alignment (not token-level KD) — aligns reasoning spaces.
- **Perspectives**: 4 by default (semantic, intent, attribute, long-tail); extensible.
