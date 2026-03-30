from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import load_tasks
from model import TemplateSelector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    tasks = load_tasks(args.train)
    X = torch.tensor([t.spec.to_features() for t in tasks], dtype=torch.float32)
    y = torch.tensor([t.label for t in tasks], dtype=torch.long)

    model = TemplateSelector()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(X.shape[0])
        total_loss = 0.0
        n = 0

        for i in range(0, X.shape[0], args.batch_size):
            idx = perm[i : i + args.batch_size]
            logits = model(X[idx])
            loss = torch.nn.functional.cross_entropy(logits, y[idx])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach())
            n += 1

        with torch.no_grad():
            model.eval()
            logits = model(X)
            acc = (logits.argmax(dim=-1) == y).float().mean().item()

        print(f"epoch={ep} loss={total_loss/max(1,n):.4f} acc={acc:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
