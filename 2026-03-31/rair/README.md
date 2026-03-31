# RAIR (toy reproduction)

Toy reproduction skeleton for:

- **RAIR: A Rule-Aware Multimodal Benchmark for Challenging E-Commerce Relevance Assessment** (arXiv:2512.24943)

## Idea captured

- Synthetic (query, item) relevance labels with an explicit **rule checklist**.
- 4-level relevance labels **L1–L4** derived deterministically from rule satisfaction.
- Three subsets: **General / Hard / Visual-Salient**.
- Evaluation metrics: **Acc@4**, **Acc@2** (L1/L2 vs L3/L4), **Macro-F1**.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-31/rair
python3 -m pip install -r requirements.txt
python3 train.py --epochs 8 --use_rules
python3 test.py --use_rules
```

## Notes

This is a toy benchmark. The goal is to provide a runnable evaluation pipeline and demonstrate how rules can be injected as features.
