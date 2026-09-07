# TradeUpDistill (Toy Reproduction)

Lightweight PyTorch reproduction for **Distill Globally, Adapt Locally: Reasoning Distillation and Product-Type Test-Time Training for Scalable Trade-Up Recommendation**.

Paper: https://arxiv.org/abs/2609.05363

## Implemented ideas

- Directional product-pair relation classification from precomputed product embeddings.
- Four-class LLM-teacher label distillation with rationale embedding alignment.
- Contrastive pair representation learning so examples with the same relation are close.
- Product-type test-time training (PT-TTT) through lightweight frozen-backbone adapters.
- Toy dataset, training script, and evaluation script with AUC/AP/F1 metrics.

## Quickstart

```bash
pip install -r requirements.txt
python3 train.py --epochs 3
python3 test.py --checkpoint outputs/tradeup_student.pt
```

## Notes

The industrial teacher LLM, Amazon catalog, and product-type support sets are not public. This reproduction keeps the interfaces aligned with the paper and uses synthetic product embeddings/rationales so the full pipeline is runnable end-to-end.
