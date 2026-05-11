# UniVA — Toy PyTorch Reproduction

Faithful-structure toy reproduction of *UniVA: Unified Value Alignment for Generative Recommendation in Industrial Advertising* (arXiv:2605.05803).

The original paper does not provide code. This reproduction implements the three pillar components end-to-end on a toy synthetic ad-recommendation dataset:

1. **Commercial SID Tokenizer** — RQ-VAE-style hierarchical tokenizer that injects value attributes (eCPM bucket, category margin) into SID code-book construction.
2. **Generation-as-Ranking SID Decoder** — autoregressive decoder jointly trained with SFT (next-SID-token CE) and an eCPM-aware reward; logits double as ranking scores.
3. **Value-Guided Personalized Beam Search** — beam search constrained by a per-user trie of valid SID paths and re-weighted by value scores at each step.

Real eCPM RL is replaced with a toy reward signal so the pipeline is trainable on CPU. Production-grade pieces (full RL trainer, online trie merge, online value oracle) are marked `# TODO[paper]:` with the relevant equations / pseudocode.

## Files

| File | Purpose |
|------|---------|
| `model.py` | Tokenizer + decoder + value-guided beam search |
| `data.py` | Synthetic toy ad-rec dataset with value attributes |
| `train.py` | SFT + value-aware RL training loop |
| `infer.py` | Inference / evaluation entry point |
| `requirements.txt` | torch + numpy |

## Quick start

```bash
pip install -r requirements.txt
python train.py --epochs 3 --batch-size 64
python infer.py --ckpt outputs/univa.pt
```

Toy hit@10 will be printed at the end of inference.
