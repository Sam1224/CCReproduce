from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyTradeUpDataset, split_dataset
from model import TradeUpStudent, tradeup_losses


def run_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(batch)
        loss, _ = tradeup_losses(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
    return total / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default="outputs/tradeup_student.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ToyTradeUpDataset(num_samples=2048)
    train_set, _ = split_dataset(dataset)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    model = TradeUpStudent(num_product_types=dataset.num_product_types).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, loader, optimizer, device)
        print(f"epoch={epoch} loss={loss:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "num_product_types": dataset.num_product_types}, output)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
