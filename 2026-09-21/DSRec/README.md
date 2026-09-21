# DSRec toy reproduction

PyTorch implementation of the key idea in **Dual-Interest Sequential Product Recommendation With Multi-Granular SSM**: each item has a long-term role and a short-term time-sensitive role. The demo uses a GRU as a practical full-sequence SSM proxy and a time-modulated recurrent SSM branch, then fuses the two branches with residual cross-granularity exchange.

Run:

```bash
python train.py
python test.py
```
