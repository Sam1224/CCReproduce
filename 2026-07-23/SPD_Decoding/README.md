# SPD Decoding Toy Reproduction

This folder implements a runnable PyTorch surrogate of **Stochastic Primal-Dual Decoding for Multiobjective Generative Recommender Systems**.

The original paper focuses on inference-time multi-objective control for generative recommendation. This reproduction keeps the same core pipeline while replacing proprietary industrial data with a CPU-friendly synthetic slate-generation task:

- `data.py` builds a toy catalog with item relevance, creator-group exposure, category diversity, and business scores.
- `model.py` implements a generative slate model plus a stochastic primal-dual decoder that updates dual variables during decoding.
- `train.py` trains the slate generator with teacher forcing on synthetic oracle slates.
- `test.py` compares greedy decoding against stochastic primal-dual decoding on relevance, business score, tail-creator exposure, category coverage, and constraint violations.

The exact online serving stack, theoretical regret proofs, and proprietary Spotify logging pipeline are not reproducible here. Instead, the folder preserves the method interface and exposes the same control points that matter for downstream experimentation.

Run:

```bash
pip install -r requirements.txt
python train.py --epochs 4
python test.py --checkpoint checkpoints/spd_decoding.pt
```
