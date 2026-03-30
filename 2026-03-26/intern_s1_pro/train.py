from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import make_pairs
from model import ImgEncoder, TxtEncoder, contrastive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = make_pairs(n=4000, seed=1)

    img = ImgEncoder().to(device)
    txt = TxtEncoder().to(device)
    opt = torch.optim.AdamW(list(img.parameters()) + list(txt.parameters()), lr=args.lr)

    for ep in range(args.epochs):
        img.train(); txt.train()
        loss = contrastive(img(data.img.to(device)), txt(data.txt.to(device)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"epoch={ep} loss={loss.item():.4f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"img": img.state_dict(), "txt": txt.state_dict()}, ckpt / "intern_s1_pro.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
