# DeCoS_OWVAD

Toy but runnable PyTorch reproduction for **RETHINKING OPEN-WORLD VIDEO ANOMALY DETECTION: DIAGNOSING DEFINITION BLINDNESS** (arXiv:2607.20780).

## What is reproduced

This folder keeps the paper's core logic instead of trying to rebuild the full CLIP + LaGoVAD stack:

- simulate a video clip with **shared anomaly evidence** and class-specific anomaly evidence
- expose why a query-independent anomaly prior leads to **definition blindness**
- implement the paper's **Definition-Contrastive Scoring (DeCoS)** idea via:
  - anomaly-vs-normal margin computation
  - cross-definition centering to remove shared anomaly evidence
  - a lightweight temporal residual readout with zero-sum centering
  - a definition-agnostic anomaly gate
- evaluate with definition-conditioned probes inspired by the paper:
  - standard anomaly AUROC
  - DC-Disc
  - DC-DetΔ
  - DC-SelΔ

## Files

- `dataset.py`: synthetic OWVAD clip generator and reusable definition bank
- `model.py`: definition-blind, margin-only, centered-margin, and DeCoS scorers
- `metrics.py`: toy evaluation probes
- `train.py`: trains only the DeCoS residual readout on frame-level anomaly-class supervision
- `test.py`: compares all scorers and prints a compact metric table

## Run

```bash
python train.py --epochs 12 --output decos_toy.pt
python test.py --checkpoint decos_toy.pt
```

## Expected behavior

A successful run should show the same qualitative pattern as the paper:

- `definition_blind` keeps a strong standard anomaly AUROC but weak definition-following scores
- `margin_only` improves slightly, but still leaks shared anomaly evidence
- `centered_margin` improves definition-conditioned separation
- `decos` gives the strongest `dc_disc`, `dc_det_delta`, and `dc_sel_delta`

## Important differences from the original paper

- The original paper evaluates on UCF-Crime, XD-Violence, and MSAD with frozen vision-language features and LaGoVAD-style components.
- This toy version uses **synthetic frame features** instead of real video clips and CLIP embeddings.
- The anomaly gate is a frozen linear scorer aligned with the shared anomaly direction, replacing the original pre-trained VAD detector.
- The residual head is a tiny temporal Conv1D block rather than the paper's full readout stack.
- The evaluation formulas are simplified but preserve the same diagnostic intent: standard anomaly detection can look good even when the queried definition does not change the ranking.

## Why this toy setup is still useful

The paper's central finding is structural: **shared anomaly evidence can dominate definition-conditioned scoring**. This reproduction keeps that failure mode explicit and demonstrates why subtracting the shared component is effective, which makes it suitable for downstream experimentation in content governance, video moderation, and evolving policy definitions.
