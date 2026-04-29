# IECD[2] — Instruction–Evidence Contrastive Dual-Stream Decoding (Reproduction)

This folder reproduces the **core decoding math** of IECD[2] from:

- **Instruction-Evidence Contrastive Dual-Stream Decoding for Grounded Vision-Language Reasoning** (arXiv:2604.25809)

The original paper proposes an inference-time decoding method for instruction-tuned VLMs to reduce hallucination by balancing:
- an **instruction stream** (fluent / informative, but prone to language priors), and
- an **evidence stream** (strictly grounded, but conservative).

The two streams are fused token-by-token via a **symmetric-KL gated geometric mean**.

## What’s implemented

- `iecd2/core.py`: temperature scaling, symmetric KL divergence, gate `g = sigmoid(eta * D)`, and fused distribution `p ∝ p_I^g * p_E^(1-g)`.
- `iecd2/hf_adapter.py`: a minimal HuggingFace **text-only** LM adapter so the code can be executed on CPU with small models.
- `scripts/demo_text_lm.py`: demonstrates the fusion on a toy prompt pair.
- `tests/test_core.py`: unit tests for gating behavior.

## What’s NOT implemented (and why)

A full end-to-end reproduction on **LLaVA / InstructBLIP** would require:
- multi-modal input formatting (prompt + image tensor),
- per-stream KV-cache management for efficient step-by-step decoding,
- evaluating on POPE/MME/VQAv2/AMBER with the same protocols.

These are engineering-heavy and GPU-dependent; this repo focuses on faithfully reproducing IECD’s **algorithmic core** in a model-agnostic way.

## Quickstart

```bash
cd 2026-04-29/IECD2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Demo on a small CPU model
python scripts/demo_text_lm.py --model distilgpt2 --question "Describe the image." --eta -3

# Run tests
pytest -q
```

## How to extend to a real VLM

Replace `HFTextLM` with a VLM adapter that exposes:

- `next_token_logits(prompt: str, image: PIL.Image | torch.Tensor, past_key_values=...) -> logits`

Then call `iecd_fuse_logits(logits_i, logits_e, config)` at every decoding step.
