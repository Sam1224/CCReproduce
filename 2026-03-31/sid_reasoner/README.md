# SIDReasoner (reproduction-oriented implementation)

Reference paper: **Reasoning over Semantic IDs Enhances Generative Recommendation** (arXiv:2603.23183)

This folder implements a runnable SIDReasoner-style pipeline with two stages:

1) **SID–language alignment via multi-task SFT** on a mixed corpus where sequences contain both:
   - Semantic IDs (SIDs) representing items
   - natural-language descriptions / constraints

2) **Outcome-driven GRPO-style RL** that samples K reasoning trajectories per context and optimizes reward based on:
   - prediction correctness (matching target SID)
   - structured output format reward

Because the original paper uses large pretrained LLMs + teacher-synthesized enriched corpora, this reproduction:

- implements the **full training logic** (SFT + GRPO) and model architecture (causal Transformer LM)
- uses a **synthetic but non-trivial** generative recommendation dataset with SIDs
- clearly marks what requires a large teacher model as pseudocode

## Quickstart

```bash
cd ccreproduce_repo/2026-03-31/sid_reasoner
python3 -m pip install -r requirements.txt
python3 train.py --stage sft --epochs 3
python3 train.py --stage rl  --steps 300 --rollouts 16
python3 eval.py --ckpt checkpoints/sid_reasoner.pt
```

## Pseudocode (teacher-synthesized enriched corpus)

```text
# NOT IMPLEMENTED
# Use a stronger teacher LLM to synthesize diverse SID-language pairs
for item in catalog:
  prompt = make_prompt(item_metadata, user_history)
  enriched_text = TeacherLLM(prompt)
  add_to_alignment_corpus(enriched_text)
```
