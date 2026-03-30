from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import make_dataset, split
from model import EgoEncoder


def acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = make_dataset(n=4000, seed=1)
    tr, te = split(batch)

    model = EgoEncoder(vocab=64, num_tasks=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        logits = model(tr.seq.to(device))
        loss = torch.nn.functional.cross_entropy(logits, tr.y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            a = acc(model(te.seq.to(device)).cpu(), te.y)
        print(f"epoch={ep} loss={loss.item():.4f} acc={a:.3f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, ckpt / "ego2web.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
