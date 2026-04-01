from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ToyMemRerankDataset
from model import MemRerankToyModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", default="ckpt.pt")
    args = ap.parse_args()

    device = torch.device("cpu")

    train_ds = ToyMemRerankDataset(n=30000, seed=0)
    val_ds = ToyMemRerankDataset(n=6000, seed=1)

    model = MemRerankToyModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            pos, neg = model(
                history_item_ids=batch.history_item_ids.to(device),
                query_token_ids=batch.query_token_ids.to(device),
                pos_item_id=batch.pos_item_id.to(device),
                neg_item_id=batch.neg_item_id.to(device),
            )
            # Pairwise logistic loss.
            loss = F.softplus(-(pos - neg)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                pos, neg = model(
                    history_item_ids=batch.history_item_ids.to(device),
                    query_token_ids=batch.query_token_ids.to(device),
                    pos_item_id=batch.pos_item_id.to(device),
                    neg_item_id=batch.neg_item_id.to(device),
                )
                correct += (pos > neg).sum().item()
                total += pos.numel()
        acc = correct / max(1, total)
        print(f"epoch={epoch} pairwise_acc={acc:.4f}")

    out = Path(args.out)
    torch.save({"model": model.state_dict()}, out)


if __name__ == "__main__":
    main()
