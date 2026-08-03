from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import PRIMITIVES, create_dataloaders
from model import EvoReason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default="evoreason_toy.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catalog, train_loader, test_loader = create_dataloaders(batch_size=args.batch_size)
    model = EvoReason(
        num_items=catalog.item_features.size(0),
        num_primitives=len(PRIMITIVES),
        feature_dim=catalog.item_features.size(1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    item_features = catalog.item_features.to(device)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_recall = 0.0
        running_primitive_acc = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, metrics = model.loss(batch["history"], batch["target"], batch["primitive_labels"], item_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_recall += metrics["recall@1"]
            running_primitive_acc += metrics["primitive_acc"]
        print(
            f"epoch={epoch + 1} loss={running_loss / len(train_loader):.4f} "
            f"recall@1={running_recall / len(train_loader):.4f} primitive_acc={running_primitive_acc / len(train_loader):.4f}"
        )

    metrics = {"recall@1": 0.0, "recall@5": 0.0, "primitive_acc": 0.0}
    model.eval()
    for batch in test_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_metrics = model.evaluate(batch["history"], batch["target"], batch["primitive_labels"], item_features)
        for key, value in batch_metrics.items():
            metrics[key] += value
    metrics = {key: value / len(test_loader) for key, value in metrics.items()}
    print(f"[eval] {metrics}")

    checkpoint_path = Path(args.checkpoint)
    torch.save({"model_state": model.state_dict(), "item_features": catalog.item_features}, checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
