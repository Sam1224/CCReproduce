# ARGUS — Reproduction

**Paper:** ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement with Adversarial Umpiring  
**arXiv:** https://arxiv.org/abs/2605.02200  
**Authors:** Deyi Ji, Junyu Lu, Xuanyi Liu, Liqun Liu, Hailong Zhang, Peng Shu, Huan Yu, Jie Jiang, Tianru Chen, Lanyun Zhu

---

## Overview

ARGUS addresses the **non-stationary policy problem** in online advertising governance.
When regulatory policies change (e.g., new restrictions on "educational anxiety ads"),
historical labels conflict with new mandates, and naive fine-tuning fails.

### Three-Stage Framework

```
Stage I: Policy Seeding
    ─── Few new-policy samples → initial policy perception
    ─── Outcome: baseline model M₀ aware of new policy

Stage II: Adversarial Label Rectification
    ─── Prosecutor: argues violation (supports new policy)
    ─── Defender:   argues compliance (supports historical label)
    ─── Umpire:     arbitrates → corrected label
    ─── Outcome: cleaned dataset D_rect

Stage III: Latent Knowledge Discovery
    ─── Tripartite dialectical discussion on gray-area samples
    ─── Hard adversarial mining: synthetic edge cases
    ─── Outcome: augmented dataset D_final, model M_final
```

## Files

```
ARGUS/
├── README.md
├── requirements.txt
├── data/
│   └── toy_ad_governance.jsonl    # Toy ad samples with labels + policy versions
├── stages/
│   ├── stage1_seeding.py          # Stage I: Policy Seeding
│   ├── stage2_rectification.py    # Stage II: PDU adversarial rectification
│   └── stage3_discovery.py        # Stage III: Latent Knowledge Discovery
├── models/
│   └── ad_classifier.py           # Ad governance classifier (shared encoder)
├── pdu/
│   └── prosecutor_defender_umpire.py  # Core PDU architecture
├── pipeline.py                    # Full ARGUS training pipeline
└── evaluate.py                    # Evaluation on historical + new-policy data
```

## Quick Start

```bash
pip install -r requirements.txt

# Run full ARGUS pipeline
python pipeline.py --data data/toy_ad_governance.jsonl

# Evaluate policy-adaptive performance
python evaluate.py --model_dir outputs/ --data data/toy_ad_governance.jsonl
```
