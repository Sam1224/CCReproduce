# SAGER (Toy Reproduction)

Paper: **SAGER: Self-Evolving User Policy Skills for Recommendation Agent** (arXiv:2604.14972)

This toy implementation reproduces the *core abstraction* (not the full LLM pipeline):

- per-user **policy skill** that evolves with interaction feedback
- **full skill** vs **slim skill** (distilled runtime representation)
- incremental update in the spirit of **reinforce / discover / weaken**
- listwise ranking driven by the slim skill

It uses a synthetic item-attribute world (no LLM calls) so the evolution logic is runnable and testable.

## Quickstart

```bash
pip install torch
python train.py
python test.py
```
