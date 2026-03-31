# BLM-Guard (toy reproduction)

Toy reproduction skeleton for:

- **BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards** (arXiv:2602.18193, AAAI 2026)

## Idea captured

This repo implements a runnable *toy* pipeline capturing the paper’s shape:

- **Risk-prompt-anchored keyframe selection** with BIN+TOP.
- **Patch/region saliency** selection (simplified).
- **Multi-task moderation head**: predict (scene, type) and a binary risky decision.
- **Policy-aligned training**: supervised warmup + simplified GRPO-style RL with a hybrid reward:
  - `rrule`: scene/type correctness
  - `rformat`: whether a `<think>`/`<answer>` formatted output is produced
  - `rscaR`: a simple self-consistency score between rationale and decision

## Quickstart

```bash
cd ccreproduce_repo/2026-03-31/blm_guard
python3 -m pip install -r requirements.txt
python3 train.py --epochs 6 --rl_steps 300
python3 test.py
```

## Notes

- This is **not** a full VLM/LLM reproduction.
- The rationale is template-generated to keep the toy pipeline runnable.
