# WHALE (toy reproduction)

This toy implementation mirrors the paper's core idea: a unified recommendation model that combines non-sequential high-order feature interaction (Wukong-style), long user behavior sequence modeling (HSTU-style), and a fusion module that lets static cross features query temporal intent.

## Files
- `data.py`: synthetic recommendation dataset with user, item, context, and behavior sequence.
- `model.py`: Wukong block, HSTU block, fusion module, and binary ranking head.
- `train.py`: train the toy model end-to-end.
- `test.py`: load a checkpoint and report recommendation metrics.

## Run
```bash
python3 train.py --epochs 5
python3 test.py --checkpoint checkpoints/whale.pt
```

## Notes
- The dataset is synthetic, but field interfaces are aligned with a recommendation pipeline.
- The implementation keeps the paper's architectural decomposition while simplifying industrial-scale infrastructure.
