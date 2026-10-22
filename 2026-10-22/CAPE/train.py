from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ToySequentialDataset, collate_batch
from model import SemanticIdRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/backbone.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ToySequentialDataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    model = SemanticIdRecommender(dataset.num_items).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            history = batch["history"].to(args.device)
            target = batch["target"].to(args.device)
            loss = loss_fn(model(history), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * history.size(0)
        print(f"epoch={epoch} loss={total_loss / len(dataset):.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "num_items": dataset.num_items,
            "head_cutoff": dataset.head_cutoff,
        },
        args.output,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

