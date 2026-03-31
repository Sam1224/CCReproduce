# AwaRes (toy reproduction)

This folder is a **lightweight, runnable reproduction skeleton** of the core idea in:

- **Look Where It Matters: High-Resolution Crops Retrieval for Efficient VLMs (AwaRes)** (arXiv:2603.16932)

## What this reproduction covers

AwaRes proposes a *spatial-on-demand* inference pattern: a model first answers from a low-resolution global view, and only when needed it issues a tool call to fetch a small set of high-resolution crops, trading off accuracy vs. compute.

The original paper targets large VLMs, automatic supervision via low-vs-high judges, and multi-turn GRPO optimization. Here we implement a faithful *mechanism-level* toy version:

- Synthetic high-res images contain a small “digit stamp” placed at a random location.
- A low-res global view is obtained by downsampling.
- The policy chooses between **NO_CROP** and **GET_CROP[idx]** (a discrete 3×3 crop set).
- If a crop is requested, the environment returns the corresponding high-res crop (downsampled to the same resolution as the global view).
- Training: cold-start supervised imitation (SFT) + a simple REINFORCE-style RL fine-tuning with a reward that penalizes unnecessary crops.

## Quickstart

```bash
cd 2026-03-24/awares
python3 -m pip install -r requirements.txt
python3 train.py
python3 test.py
```

## Files

- `dataset.py`: synthetic image generation + crop tool
- `model.py`: tiny “AwaRes” policy + answer heads
- `train.py`: SFT + RL fine-tuning demo
- `test.py`: smoke test

## Notes / limitations

- This is **not** a full VLM reproduction (no OCR, no real vision-text backbone, no GRPO).
- It is intended as a clean reference for the *multi-turn tool-calling crop retrieval* pattern and the *accuracy–efficiency* objective.
