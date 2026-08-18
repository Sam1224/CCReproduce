# TaoLiveHAT

Toy PyTorch reproduction for **TaoLive Digital Avatar Agent Technical Report: Training Agents to Evolve with Their Harness**.

## What is implemented

This folder focuses on the paper's core idea: training a compact live-commerce policy model that remains robust when the execution harness changes.

- `data.py`: generates a toy live-commerce QA dataset with semantic features and evolving harness states.
- `model.py`: a lightweight harness-aware policy network.
- `train.py`: trains two variants, `fixed_harness` and `hat`.
- `test.py`: evaluates base QA accuracy, harness-variant robustness, and average inference latency.

## Run

```bash
python3 train.py
python3 test.py
```

Artifacts are written to `artifacts/`.

## Mapping to the paper

- **Harness-State Augmentation (HSA)** is approximated by sampling perturbed prompt, tool, and hook representations.
- **Fixed-harness SFT vs HAT** is approximated by training one model on canonical harness states and another on augmented harness states.
- **Live-room simulator / RL** is simplified into a classification-style policy objective because the public paper does not disclose a runnable simulator.

## Not implemented

The full industrial simulator, long-horizon RL environment, and production serving stack are not publicly available, so they are described here rather than fully reproduced.
