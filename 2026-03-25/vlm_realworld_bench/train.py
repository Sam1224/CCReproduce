from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import make_dataset, split
from model import FusionModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    train, test = split(make_dataset(n=6000, seed=1, ood=False), 0.85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Answers: boolean(2) or index(6) or count(6) => take max 6
    model = FusionModel(img_dim=train.img.shape[1], q_vocab=4, num_answers=6).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        logits = model(train.img.to(device), train.q.to(device))
        loss = torch.nn.functional.cross_entropy(logits, train.y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(test.img.to(device), test.q.to(device)).argmax(dim=-1).cpu()
        acc = (pred == test.y).float().mean().item()
        print(f"epoch={ep} loss={loss.item():.4f} acc={acc:.3f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "img_dim": int(train.img.shape[1])}, ckpt / "vlm_realworld.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
