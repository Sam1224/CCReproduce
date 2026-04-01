from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ToyModerationDataset
from model import XuanwuToyMMClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="ckpt.pt")
    args = ap.parse_args()

    device = torch.device("cpu")

    train_ds = ToyModerationDataset(n=6000, seed=0)
    val_ds = ToyModerationDataset(n=1200, seed=1)

    model = XuanwuToyMMClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            logits = model(image_feat=batch.image_feat.to(device), token_ids=batch.token_ids.to(device))
            loss = F.cross_entropy(logits, batch.label.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(image_feat=batch.image_feat.to(device), token_ids=batch.token_ids.to(device))
                pred = logits.argmax(dim=-1)
                correct += (pred.cpu() == batch.label).sum().item()
                total += batch.label.numel()
        acc = correct / max(1, total)
        best = max(best, acc)
        print(f"epoch={epoch} val_acc={acc:.4f}")

    out = Path(args.out)
    torch.save(
        {
            "model": model.state_dict(),
            "best_val_acc": best,
        },
        out,
    )


if __name__ == "__main__":
    main()
