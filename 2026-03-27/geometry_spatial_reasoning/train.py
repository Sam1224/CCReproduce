from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import make_dataset, split
from model import BaselineMLP, GeometryAware, acc


def train(model, tr, te, epochs: int, lr: float, name: str) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        logits = model(tr.pts, tr.q)
        loss = torch.nn.functional.cross_entropy(logits, tr.y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            a = acc(model(te.pts, te.q), te.y)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"{name} ep={ep} loss={loss.item():.4f} acc={a:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    tr, te = split(make_dataset(n=9000, seed=1), 0.85)

    base = BaselineMLP()
    geo = GeometryAware()

    train(base, tr, te, epochs=args.epochs, lr=3e-3, name="baseline")
    train(geo, tr, te, epochs=args.epochs, lr=3e-3, name="geometry")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"baseline": base.state_dict(), "geometry": geo.state_dict()}, ckpt / "geometry.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
