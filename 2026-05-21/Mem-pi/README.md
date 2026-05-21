# Mem-π (Toy Reproduction)

- Paper: **Mem-π: Adaptive Memory through Learning When and What to Generate**
- arXiv: https://arxiv.org/abs/2605.21463
- Authors: Xiaoqiang Wang, Chao Wang, Hadi Nekoei, Christopher Pal, Alexandre Lacoste, Spandana Gella, Bang Liu, Perouz Taslakian
- Affiliation: ServiceNow AI Research; Mila – Quebec AI Institute; Université de Montréal; Polytechnique Montréal; McGill University; CIFAR

## What is reproduced

This folder provides a **minimal, runnable PyTorch reproduction** of the *core algorithmic idea* in Mem-π:

1. A **separate parametric memory policy** \(\pi_{mem}\) (implemented as `MemPiPolicy`) that decides:
   - **when** to generate memory (`[ABSTAIN]` vs `[GENERATE]`), and
   - **what** memory to generate (here discretized as a tool-id hint).
2. A **two-stage training recipe**:
   - **Stage 1 – Experience distillation**: supervised learning from an offline experience bank.
   - **Stage 2 – Adaptation distillation**: a toy variant of the paper’s *decision–content decoupled policy optimization* using a structured rollout group (one abstain branch + multiple generate branches).

## Toy environment

We use a tiny “tool-use” environment inspired by web-agent tasks:

- Each task has a hidden `tool_id` that must be selected.
- A weak **base agent** can solve tasks that include an explicit clue; otherwise it defaults to tool 0.
- The **memory policy** generates a hint (tool id) only when it increases downstream task success.

This setup is intentionally simple so you can run end-to-end training quickly while still observing the key behavior: **adaptive abstention + useful memory generation**.

## Quickstart

```bash
python -m pip install -r requirements.txt

# Train (Stage1 + Stage2) and save checkpoints
python train.py --device cpu --out checkpoints

# Evaluate a trained checkpoint
python eval.py --ckpt checkpoints/stage2.pt
```

Expected outcome: `adaptive_sr` should be noticeably higher than `base_sr`, and `avg_generate_prob` should converge to a value < 1.0 (i.e., the policy learns to abstain on easy cases).

## Notes & limitations

- This is **not** a full reproduction of the paper’s LLM/VLM memory model training pipeline (WebArena/WorkArena/ALFWorld/LAB), nor the GRPO/PPO implementation details.
- The goal is to provide a clean reference implementation of the **decision vs content decoupling idea** with a complete data/model/train/eval pipeline.
