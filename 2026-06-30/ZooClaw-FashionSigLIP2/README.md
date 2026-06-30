# ZooClaw-FashionSigLIP2 — Distilled Fine-tuning + WiSE-FT (Toy Reproduction)

- Paper: **ZooClaw-FashionSigLIP2: Distilled Fine-tuning for Robust Fashion Retrieval**
- arXiv: https://arxiv.org/abs/2606.27708

## What is reproduced

The paper studies a common *domain adaptation trade-off* for vision-language retrieval: aggressively fine-tuning a foundation dual-encoder boosts in-domain (ID) fashion retrieval but may hurt out-of-distribution (OOD) robustness.

The paper’s recipe is:

1. **Full fine-tuning with knowledge distillation** on curated in-domain fashion data.
2. **WiSE-FT** weight interpolation between the fine-tuned model and the original base model to recover robustness.

This folder provides a **toy, runnable PyTorch reproduction** that keeps the same *training/evaluation structure*:

- A dual-encoder (image encoder + text encoder) trained with symmetric contrastive loss.
- A deterministic **teacher** that provides similarity targets for distillation.
- An explicit **OOD split** to demonstrate robustness collapse & WiSE-FT recovery.
- A benchmark-style evaluation printing **R@K** for both short queries and long queries.

> Note: This is not a faithful reproduction of SigLIP2’s backbone nor ZooClaw’s private training corpus. It is a pipeline-aligned reproduction focused on the method logic (distillation + WiSE-FT) rather than scale.

## Files

- `data.py`: toy fashion items, synthetic image rendering, tokenizer, teacher embedder
- `model.py`: dual-encoder + contrastive loss + distillation KL + WiSE-FT interpolation
- `train.py`: (1) foundation pre-training (broad) then (2) distilled full fine-tuning (curated)
- `eval.py`: ID & OOD retrieval evaluation (R@1/5/10) + WiSE-FT sweep
- `run_pipeline.py`: runs `train.py` then `eval.py`

## Quickstart

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Expected output (example)

You should see:

- In-domain short-query R@K improves after fine-tuning.
- OOD short-query R@K may drop after fine-tuning.
- A WiSE-FT `alpha` in (0, 1) often improves OOD R@10 compared to pure fine-tuned weights.

## Mapping to the paper

- **Distilled full fine-tuning**: `train.py` Stage 1 (`[finetune+distill]`) uses contrastive + KL distillation.
- **WiSE-FT**: `eval.py` interpolates weights between `base.pt` and `finetuned.pt`.

## Limitations

- The teacher is a deterministic attribute embedder rather than a large VLM.
- The dataset is synthetic; metrics are meant to be *structural* (pipeline sanity) instead of SOTA.
- For real SigLIP2/ZooClaw training, replace `TinyImageEncoder/TinyTextEncoder` with a pretrained backbone and swap the dataset loader.
