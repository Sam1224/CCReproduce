# ProductConsistency (toy reproduction)

Faithful, self-contained PyTorch reproduction of the **method logic** in:

> Khanna, Yadav, Singh. *ProductConsistency: Improving Product Identity Preservation
> in Instruction-Based Image Editing via SFT and RL.* arXiv:2606.19103 (2026-06-17).

The paper fine-tunes large diffusion editors (Qwen-Image-Edit-2511 / Flux.1-Kontext-dev)
and its official code is hosted on an anonymous double-blind site whose completeness /
runnability cannot be verified. This folder therefore reproduces the core ideas at toy
scale while keeping every interface aligned (image tensors, description dicts).

## What is reproduced (1:1 with the paper's logic)

- **Identity-preserving instruction-based editing**: re-render the *same* product
  (brand + on-product text) onto a *new* background style requested by an instruction.
- **Two-stage training**: SFT (reconstruct GT edited target) → RL.
- **Cyclic Consistency reward** (the paper's key contribution): description → edited
  image → re-caption → similarity to the original description. Implemented in
  `model.cyclic_consistency_reward` and optimized with REINFORCE + baseline in `train.py`.
- **Evaluation** mirroring the paper's metrics: brand-identity accuracy, on-product text
  fidelity (OCR-style CER), background-change rate; and the **SFT vs SFT+RL** comparison
  showing RL reduces text CER (paper reports ~5× on Qwen-Image-Edit-2511).

## What is simplified (and why)

| Paper | Here |
|---|---|
| Diffusion editor (Qwen-Image-Edit / Flux) | small conv encoder/decoder editor w/ a stochastic background-style latent |
| Closed-source captioner + OCR | `ProductCaptioner` (brand head + per-cell char head) |
| 87k SFT / 869 RL real product images | `ToyProductDataset` (synthetic brand colour + digit "text" cells) |
| MLLM judge on aesthetics | omitted (left as a note; reward focuses on identity + visual consistency) |

The product/branding/text rendering is a deliberate proxy; the **reward, training loop,
and evaluation protocol are faithful** and can be swapped onto a real editor + captioner.

## Run

```bash
pip install -r requirements.txt
python train.py            # captioner warmup -> SFT -> RL (CPU friendly)
python test.py             # benchmark: SFT vs SFT+RL
```

Outputs `captioner.pt`, `editor_sft.pt`, `editor_rl.pt` and prints the benchmark table.

## Files

- `model.py`  — `ProductEditor`, `ProductCaptioner`, `cyclic_consistency_reward`
- `data.py`   — `ToyProductDataset` + renderer (shared description interface)
- `train.py`  — captioner warmup → SFT → RL (Cyclic Consistency)
- `test.py`   — ProductConsistency-style benchmark + SFT/SFT+RL comparison
