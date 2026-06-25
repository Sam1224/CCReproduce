# StreamProbeModeration (Toy Reproduction)

Toy-but-runnable reproduction for:

- **Stop Early, Spend Less: Hidden-State Probes as a Practical Recipe for Streaming Moderation of LLM Outputs** (arXiv:2606.10487v1)

## Goal

Replicate the *core idea* in a minimal setting:

- A frozen generator produces hidden states during decoding.
- A lightweight probe reads a mid-layer hidden state and predicts a safety label.
- Moderation can run **streaming / token-level** and can **stop early**.

This repo does **not** reproduce the paper's large-scale datasets or Qwen/vLLM integration. Instead it provides an end-to-end runnable pipeline (data / model / train / eval) that matches the method logic.

## Quickstart

```bash
pip install -r requirements.txt
python train_probe.py
python eval_probe.py
```

## What you should see

- Probe F1 should be high on the synthetic dataset.
- The `eval_probe.py` report includes an *early detection* metric (how many tokens after the first unsafe trigger the probe raises an alert).

## Files

- `data.py`: synthetic streaming moderation dataset.
- `model.py`: tiny frozen causal Transformer + probe.
- `train_probe.py`: trains probe on hidden states.
- `eval_probe.py`: evaluates F1 + early detection.
