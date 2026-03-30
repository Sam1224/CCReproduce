from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import make_dataset, split
from model import GraphReranker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    data = make_dataset(n=2500, seed=1)
    tr, te = split(data, 0.85)

    model = GraphReranker(d=64)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for ex in tr[:800]:
            logits = model(ex.q, ex.docs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ex.y.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n += 1
        print(f"epoch={ep} loss={total/max(1,n):.4f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, ckpt / "grapher.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
