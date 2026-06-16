# CapRL++: RLVR for Dense Image and Video Captioning

Toy reproduction of **"CapRL++: Unified Reinforcement Learning with Verifiable Rewards for Dense Image and Video Captioning"** (arXiv:2606.09393, InternLM/Shanghai AI Lab).

Official code: https://github.com/InternLM/CapRL

## Key Idea

```
Image/Video ──► [LVLM captioner] ──► Caption
                                        │
                                ┌───────▼────────┐
                                │ Vision-free LLM│
                                │  answers MCQs  │
                                └───────┬────────┘
                                        │
                              Answer Accuracy = Reward
                                        │
                              RL gradient back to captioner
```

**Caption quality = MCQ answer accuracy (utility-based, reference-free)**

## Files

- `captioner.py` — toy LVLM captioner (VLM backbone simulation)
- `mcq_scorer.py` — MCQ-based verifiable reward function
- `rl_trainer.py` — RLVR training loop (REINFORCE over caption policy)
- `data.py` — toy dataset (images → captions → MCQs)
- `evaluate.py` — Prism-style caption quality evaluation
- `requirements.txt` — dependencies

## Quick Start

```bash
pip install -r requirements.txt
python data.py            # Generate toy dataset
python rl_trainer.py      # Train with RLVR
python evaluate.py        # Evaluate caption quality
```

## Paper Reference

arXiv: https://arxiv.org/abs/2606.09393  
ICLR 2026: CapRL (predecessor) — https://github.com/InternLM/CapRL

**Key result:** 3B model matches/exceeds Qwen2.5-VL-72B on Prism captioning evaluation
