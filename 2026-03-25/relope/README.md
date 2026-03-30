# relope (toy reproduction)

This folder is a **lightweight, runnable reproduction skeleton** of the core idea in:

- **ReLope: KL-Regularized LoRA Probes for Multimodal LLM Routing** (arXiv:2603.24787)

## What this reproduction covers

The paper’s headline idea is to do *routing* (choosing which expert/model to use) with **parameter-efficient LoRA probes**, while using **KL regularization** so probes stay close to a base model distribution.

This toy implementation keeps the *mechanism*:

- Train a small **base multimodal classifier** on mixed data.
- For each “expert/domain”, train a **LoRA probe** (base frozen) on domain data.
- Add **KL(probe || base)** regularization to prevent probe over-drifting.
- Do routing by picking the probe that yields the **highest confidence / best NLL improvement**.

## Quickstart

```bash
cd 2026-03-25/relope
python3 -m pip install -r requirements.txt
python3 train.py
```

## Files

- `toy_data.py`: synthetic multimodal dataset with domains
- `lora.py`: minimal LoRA linear layer
- `model.py`: base encoder + classifier + probe wrappers
- `train.py`: base training, probe training (with/without KL), routing eval

## Notes / limitations

- This is **not** a full multimodal LLM reproduction; it is a clean toy pipeline that demonstrates **KL-regularized LoRA probes for routing**.
- Routing is implemented as a simple confidence/NLL-based selector; swapping in pairwise/listwise scoring is straightforward.
