from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import create_dataloaders
from model import TimeRoute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", type=str, default="timeroute_toy.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catalog, train_loader, test_loader = create_dataloaders(batch_size=args.batch_size, seed=args.seed)
    catalog_tensors = {key: value.to(device) for key, value in catalog.__dict__.items()}
    model = TimeRoute(num_items=catalog.text_embeddings.size(0), max_length=train_loader.dataset.sessions.size(1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_top1 = 0.0
        for batch in train_loader:
            session = batch["session"].to(device)
            timestamps = batch["timestamps"].to(device)
            target = batch["target"].to(device)
            loss, metrics = model.loss(session, timestamps, target, catalog_tensors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_top1 += metrics["top1"]
        print(f"epoch={epoch + 1} loss={running_loss / len(train_loader):.4f} top1={running_top1 / len(train_loader):.4f}")

    model.eval()
    aggregate = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "ndcg@10": 0.0}
    for batch in test_loader:
        session = batch["session"].to(device)
        timestamps = batch["timestamps"].to(device)
        target = batch["target"].to(device)
        batch_metrics = model.evaluate(session, timestamps, target, catalog_tensors)
        for key, value in batch_metrics.items():
            aggregate[key] += value
    aggregate = {key: value / len(test_loader) for key, value in aggregate.items()}
    print(f"[eval] {aggregate}")

    checkpoint_path = Path(args.checkpoint)
    torch.save({"model_state": model.state_dict(), "catalog": catalog.__dict__}, checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
