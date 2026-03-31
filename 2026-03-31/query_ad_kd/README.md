# Context-Aware Query–Ad KD (reproduction-oriented implementation)

Reference paper: **Enhancing User Safety: Context-Aware Detection of Offensive Query-Ad Pairs in Multimodal Search Advertising** (EACL 2026 Industry)

This folder implements a *non-trivial* multimodal KD pipeline:

- **Teacher**: cross-encoder (Transformer) over `[CLS] query [SEP] ad_text [SEP]` plus image/context tokens.
- **Student**: compact dual-tower + lightweight fusion.
- **Knowledge distillation**:
  - hard label BCE
  - soft logit KL (temperature)
  - intermediate feature matching on fused representation

The provided `dataset.py` supports synthetic data (default) and JSONL input.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-31/query_ad_kd
python3 -m pip install -r requirements.txt
python3 train.py --epochs 4
python3 test.py --ckpt checkpoints/query_ad_kd.pt
```

## JSONL format (optional)

```json
{"query": "...", "ad_text": "...", "img": [...], "ctx": [...], "label": 0}
```

## Pseudocode (not implemented)

### Multimodal ad creative decoding (OCR/ASR) and safety policy constraints

```text
# NOT IMPLEMENTED
ocr = OCR(image)
asr = ASR(video)
policy = SafetyPolicyEngine(query, ad_text, ocr, asr)
```
