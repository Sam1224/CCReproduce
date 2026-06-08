# TEVI (Toy Reproduction)

- Paper: **TEVI: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignment**
- arXiv: https://arxiv.org/abs/2606.07451

## What’s implemented

This folder contains a **minimal, runnable** PyTorch reproduction of TEVI’s core idea:

1. Train a **Sparse Autoencoder (SAE)** on *image embeddings* to obtain sparse latents `z`.
2. Train a **masking module** `m = sigmoid(MLP(text_emb))` to select which SAE latents to keep for a given caption.
3. Produce a **caption-conditioned edited image embedding** `x_edit = Decoder(z ⊙ m)`.
4. Optimize with an InfoNCE-style contrastive objective to improve image↔text retrieval.

To keep the pipeline self-contained, we use a **toy multimodal dataset** (synthetic image/text embeddings) that explicitly simulates the “information imbalance” assumption in the paper: image embeddings contain extra nuisance concepts not present in the caption.

## Limitations (vs. the full paper)

- This repo **does not** train CLIP/SigLIP on CC12M or evaluate on MSCOCO/DOCCI/IIW/RoCOCO.
- The **architecture & training objective are aligned** with TEVI’s description (SAE + text-conditioned masking + contrastive training), but the backbone encoders are replaced by synthetic embedding generators.

## Quickstart

```bash
cd 2026-06-08/TEVI
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train SAE + TEVI mask module
python train.py --out_dir runs/demo

# evaluate retrieval
python test.py --ckpt_dir runs/demo
```

Notes:
- Default `--sae_mode identity` uses an identity SAE (works well for the default one-hot concept basis).
- You can switch to a learned SAE with `--sae_mode train` for non-trivial embedding spaces.

Expected output includes Recall@{1,5,10} for baseline vs TEVI on the toy retrieval task.
