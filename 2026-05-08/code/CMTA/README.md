# CMTA — Toy PyTorch Reproduction

Faithful-structure toy reproduction of *CMTA: Leveraging Cross-Modal Temporal Artifacts for Generalizable AI-Generated Video Detection* (arXiv:2605.00630).

The official repo (`hwang-cs-ime/CMTA`) is presently a placeholder (LICENSE + README only); this is an independent reproduction.

## Architecture

```
video frames ─► CLIP image encoder ─┐
                                    ├─► cross-modal alignment series s_t = cos(v_t, c_t)
caption per frame ─► CLIP text enc ─┘
                          │
                          ├──► coarse branch: GRU over s_t ─┐
                          │                                  ├─► classifier ─► real / AIGV
                          └──► fine branch: Transformer ─────┘
                                over (v_t, c_t) features
```

Frame-level captions come from BLIP in the original paper. We replace BLIP/CLIP with stub feature extractors so the pipeline trains on CPU; the call sites are left ready for HuggingFace `transformers` plug-in.

## Files

| File | Purpose |
|------|---------|
| `model.py` | Stub BLIP/CLIP features + dual-branch CMTA classifier |
| `data.py` | Synthetic two-class video tensor dataset |
| `train.py` | BCE training loop |
| `infer.py` | Per-clip detection + simple AUC |
| `requirements.txt` | torch + numpy |

## Run

```bash
pip install -r requirements.txt
python train.py --epochs 5
python infer.py --ckpt outputs/cmta.pt
```
