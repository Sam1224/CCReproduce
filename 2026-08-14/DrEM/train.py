from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import create_dataloaders
from model import DrEM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="drem_toy.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = create_dataloaders(batch_size=args.batch_size, seed=args.seed)
    model = DrEM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, metrics = model.robust_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += metrics["pair_acc"]
        print(f"epoch={epoch + 1} loss={total_loss / len(train_loader):.4f} pair_acc={total_acc / len(train_loader):.4f}")

    model.eval()
    metrics = {"pair_acc": 0.0, "gauc": 0.0}
    for batch in test_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_metrics = model.evaluate(batch)
        for key, value in batch_metrics.items():
            metrics[key] += value
    metrics = {key: value / len(test_loader) for key, value in metrics.items()}
    print(f"[eval] {metrics}")

    checkpoint_path = Path(args.checkpoint)
    torch.save({"model_state": model.state_dict(), "metrics": metrics}, checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
