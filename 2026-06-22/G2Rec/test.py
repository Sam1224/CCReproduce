"""Evaluate a trained G2Rec checkpoint on the held-out next item (Sec. 5.1).

Metrics match Table 2: Recall@1/5/10, NDCG@5/10, MRR, using the leave-one-out
protocol with 1 positive + 99 sampled negatives per user.
"""
import os
import torch

from data import make_loaders
from model import G2Rec
from train import evaluate, CKPT


def main():
    if not os.path.exists(CKPT):
        raise SystemExit(f"checkpoint {CKPT} not found; run `python train.py` first")
    ck = torch.load(CKPT, weights_only=False)
    _, valid_loader, test_loader = make_loaders(ck["seqs"], ck["num_items"])
    model = G2Rec(ck["X_full"], ck["P"], ck["C"], d_model=ck.get("d_model", 64),
                  max_items=ck["max_items"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    val = evaluate(model, valid_loader)
    test = evaluate(model, test_loader)
    print("=== G2Rec evaluation (1 pos + 99 neg) ===")
    print("valid:", {k: round(v, 4) for k, v in val.items()})
    print("test :", {k: round(v, 4) for k, v in test.items()})


if __name__ == "__main__":
    main()
