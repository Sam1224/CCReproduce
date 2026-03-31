from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import make_dataset, split
from model import MemorySparseAttention


def accuracy(model: torch.nn.Module, data, *, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ex in data:
            logits = model(ex.docs.to(device), ex.query.to(device))
            pred = int(torch.argmax(logits).item())
            correct += int(pred == int(ex.y.item()))
            total += 1
    return correct / max(1, total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    data = make_dataset(n=5000, seed=1)
    tr, te = split(data, 0.85)

    model = MemorySparseAttention(d=64, heads=4, topk=args.topk, num_classes=12).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for ex in tr[:2500]:
            logits = model(ex.docs.to(args.device), ex.query.to(args.device))
            loss = torch.nn.functional.cross_entropy(logits.view(1, -1), ex.y.view(1).to(args.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
            n += 1

        acc = accuracy(model, te[:600], device=args.device)
        print(f"epoch={ep} loss={total_loss/max(1,n):.4f} acc={acc:.3f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, ckpt / "msa.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
