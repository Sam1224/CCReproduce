# GLOBE: Structured Agentic Memory for E-commerce Product Optimization

This is a lightweight PyTorch-oriented reproduction of **GLOBE: Evolving Structured Agentic Memory for E-commerce Product Optimization in Generative Search**.

The original paper uses LLM editors/rankers. This reproduction keeps the same pipeline structure while replacing closed LLM calls with deterministic, swappable components:

- structured product listings: `title`, `features`, `description`;
- field-scoped behavior handbook with helpful/harmful/neutral statistics;
- editor that applies compatible behaviors and reports `APPLIED` behavior IDs;
- neural ranker trained on toy query-product relevance data;
- deployment/evolution loop with behavior scoring and deactivation.

## Files

- `data.py` — toy e-commerce dataset and text utilities.
- `model.py` — neural ranker, behavior handbook, editor, moderator/curator logic.
- `train.py` — end-to-end deployment/evolution training loop.
- `test.py` — smoke test for ranker + handbook evolution.

## Run

```bash
pip install torch
python train.py
python test.py
```

## Notes

The paper's Reflector and generative Ranker prompts are represented as modular Python classes. To align with the full paper, replace `RuleBasedEditor` and `NeuralRanker.score_listing` with LLM-backed implementations that keep the same inputs/outputs.
