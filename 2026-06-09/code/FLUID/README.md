# FLUID — Toy Reproduction

Reproduction of the core mechanisms from:

**FLUID: From Ephemeral IDs to Multimodal Semantic Codes for Industrial-Scale Livestreaming Recommendation**
arXiv: 2605.21832

## Key Concepts Implemented

1. **LUCID Encoder** (`lucid_encoder.py`): Cross-modal fusion + Residual Vector Quantizer (K levels of hierarchical discrete codes)
2. **FLUIDRanker** (`model.py`): ID-free late-fusion ranker that uses LUCID codes as independent tokens
3. **Staged Warmup** (`train.py`): Gradually increasing VQ loss weight for stable online incremental training
4. **Evaluation** (`evaluate.py`): Recall@K comparison vs. ID-based baseline

## Files

| File | Description |
|------|-------------|
| `data.py` | Toy data generator (simulates multimodal room features + user interactions) |
| `lucid_encoder.py` | LUCID: cross-domain multimodal encoder with Residual VQ |
| `model.py` | FLUIDRanker: ID-free late-fusion ranker |
| `train.py` | Training with staged warmup |
| `evaluate.py` | Recall@K evaluation vs. ID-based baseline |
| `requirements.txt` | Dependencies |

## Usage

```bash
pip install -r requirements.txt
python train.py       # trains FLUID with staged warmup
python evaluate.py    # evaluates Recall@K vs. ID-based baseline
```

## Paper Formulas Implemented

**Residual Quantization (LUCID codes)**:
```
z = Encoder(visual, audio, text)
r_0 = z
c_k = argmin_{e in E_k} ||r_{k-1} - e||^2
r_k = r_{k-1} - c_k
LUCID_codes = (c_1, c_2, ..., c_K)
```

**ID-free Ranking Score**:
```
score = Ranker(e_user, [CodeEmbed_1(c_1), ..., CodeEmbed_K(c_K)])
```

**Staged Warmup Training Loss**:
```
L = L_ranking + alpha(t) * L_VQ
where alpha(t) gradually increases from 0 to 0.1
```

## Limitations (Toy vs. Paper)

- Uses random features instead of actual multimodal encoders
- Simplified cross-modal fusion (paper uses a more complex architecture)
- No cross-domain training (short video + livestream joint training)
- Toy data scale (500 rooms vs. millions in production)
