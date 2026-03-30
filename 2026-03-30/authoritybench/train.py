from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import make_dataset, split
from model import AuthorityRanker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    tr, te = split(make_dataset(n=4000, seed=1), 0.85)

    model = AuthorityRanker()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for ex in tr[:1200]:
            scores = model(ex.feats.unsqueeze(0))[0]
            y = ex.label
            loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), torch.tensor([int(torch.argmax(y).item())]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n += 1
        print(f"epoch={ep} loss={total/max(1,n):.4f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, ckpt / "authoritybench.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
