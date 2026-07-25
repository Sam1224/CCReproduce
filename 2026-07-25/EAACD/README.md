# EAACD Reproduction

This folder implements a compact PyTorch reproduction of **Knowledge Injection Exists in MoE? Exploring Expert-Aware Contrast Decoding in MoE for Mitigating LLMs' Hallucinations**.

The paper's anonymous repository link (`anonymous.4open.science/r/EAACD-D388/`) returned HTTP 410 / not found during verification, so this folder provides a self-contained implementation aligned with the described method: a sparse MoE language model, expert reliability grouping from high-layer router activations, hallucination-strength estimation from lower-reliability experts, adaptive contrastive logit calibration, a toy QA dataset, training, and evaluation scripts.

## Files

- `dataset.py`: toy QA dataset and vocabulary utilities with content-governance-style factual and non-factual answers.
- `model.py`: sparse MoE layers, a toy MoE LM, and `EAACDDecoder` implementing expert-aware adaptive contrastive decoding.
- `train.py`: trains the toy MoE with language-model, factuality, and router-balance losses.
- `test.py`: runs calibrated decoding and reports first-token accuracy plus factuality diagnostics.

## Run

```bash
python train.py --epochs 20
python test.py --checkpoint runs/eaacd_toy/checkpoint.pt
```

## Fidelity notes

The full paper evaluates EAACD on large MoE LLMs and QA hallucination benchmarks. This reproduction keeps the same algorithmic interfaces but uses a small toy MoE so it can run on CPU without external model weights. Replacing `ToyMoELanguageModel` with a HuggingFace MoE model requires collecting per-layer router probabilities and expert logits, then passing them through the same reliability grouping and contrast calibration logic in `EAACDDecoder`.
