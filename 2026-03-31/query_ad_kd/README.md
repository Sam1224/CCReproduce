# Query–Ad Offensive Pair Detection (toy reproduction)

Toy reproduction skeleton for:

- **Enhancing User Safety: Context-Aware Detection of Offensive Query-Ad Pairs in Multimodal Search Advertising** (EACL 2026 Industry, ACL Anthology 2026.eacl-industry.36)

## Idea captured

- **Context-aware** classification: offensiveness is triggered by the *pairing* (query, ad).
- **Teacher–student distillation**: train a stronger teacher on a small labeled set, then pseudo-label a larger pool.
- **Graph mining**: build an ad similarity graph and propagate rare positive labels to mine more candidates.
- Report metric: **AUCPR** under heavy class imbalance.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-31/query_ad_kd
python3 -m pip install -r requirements.txt
python3 train.py --epochs_teacher 5 --epochs_student 5
python3 test.py
```

## Notes

This is a toy implementation intended to reproduce the high-level pipeline (graph mining → teacher → pseudo labels → distilled student).
