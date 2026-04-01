from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SimulaToyDataset
from model import SimulaToyModel


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="ckpt.pt")
    args = ap.parse_args()

    device = torch.device("cpu")

    train_ds = SimulaToyDataset(n=30000, seed=0, hard=False)
    val_ds = SimulaToyDataset(n=6000, seed=1, hard=False)

    model = SimulaToyModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            pred = model(batch.feats.to(device))
            loss = F.mse_loss(pred, batch.target.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch in val_loader:
                preds.append(model(batch.feats.to(device)).cpu())
                targets.append(batch.target)
        val_mse = mse(torch.cat(preds, dim=0), torch.cat(targets, dim=0))
        print(f"epoch={epoch} val_mse={val_mse:.4f}")

    out = Path(args.out)
    torch.save({"model": model.state_dict()}, out)


if __name__ == "__main__":
    main()
