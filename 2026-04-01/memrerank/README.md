# MemRerank (toy reproduction)

Paper: **MemRerank: Preference Memory for Personalized Product Reranking** (arXiv:2603.29247)

This is a simplified PyTorch reproduction that demonstrates the core idea:

- distill user history into a **query-independent preference memory**
- rerank candidates by `score = <item, query_repr + alpha * memory>`

## Run

```bash
python3 train.py --epochs 2
python3 test.py --ckpt ckpt.pt
```

## Notes

- This toy dataset uses a categorical preference rule to simulate purchase history and preferred items.
- It is meant to provide a runnable end-to-end pipeline (data/model/train/test) rather than a full-scale industrial system.
