from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import make_dataset, split
from model import MemoryReader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    tr, te = split(make_dataset(n=6000, seed=1), 0.85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MemoryReader().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        logits = model(tr.facts.to(device), tr.query.to(device))
        loss = torch.nn.functional.cross_entropy(logits, tr.answer.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(te.facts.to(device), te.query.to(device)).argmax(dim=-1).cpu()
        acc = (pred == te.answer).float().mean().item()
        print(f"epoch={ep} loss={loss.item():.4f} acc={acc:.3f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, ckpt / "memorycd.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
