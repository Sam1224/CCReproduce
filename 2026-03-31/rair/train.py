from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import make_dataset, split
from model import RAIRModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--use_rules", action="store_true")
    args = ap.parse_args()

    data = make_dataset(n=12000, seed=0)
    tr, _ = split(data, 0.85)

    model = RAIRModel(use_rules=args.use_rules)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for ex in tr[:3000]:
            logits = model(
                torch.tensor([ex.q_cat]),
                torch.tensor([ex.q_brand]),
                torch.tensor([ex.q_color]),
                torch.tensor([ex.item_cat]),
                torch.tensor([ex.item_brand]),
                torch.tensor([ex.item_color_text]),
                torch.tensor([ex.item_color_img]),
                ex.rule_ids.unsqueeze(0),
            )
            loss = F.cross_entropy(logits, torch.tensor([ex.y]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n += 1
        print(f"epoch={ep} loss={total/max(1,n):.4f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "use_rules": args.use_rules}, ckpt / "rair.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
