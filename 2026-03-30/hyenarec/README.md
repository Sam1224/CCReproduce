# Hyena Operator SeqRec (toy reproduction)

Toy reproduction skeleton for:

- **Hyena Operator for Fast Sequential Recommendation** (arXiv:2603.25027)

## Idea captured

We implement a simple sequential recommendation baseline where a “Hyena-like” long convolution replaces attention:

- User history is a sequence of item IDs.
- Model uses embedding + depthwise 1D convolution blocks to capture long-range dependencies efficiently.
- Predict next-item via softmax over a small item vocab.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-30/hyenarec
python3 -m pip install -r requirements.txt
python3 train.py --epochs 5
python3 test.py
```
