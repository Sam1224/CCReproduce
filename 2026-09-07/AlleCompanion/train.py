from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset import ToyComplementDataset
from model import TwoTower, companion_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=str, default="outputs/allecompanion.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ToyComplementDataset()
    train_set, _ = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))], generator=torch.Generator().manual_seed(5))
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    model = TwoTower(feature_dim=dataset.feature_dim, num_categories=dataset.num_categories).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, _ = companion_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        print(f"epoch={epoch} loss={total / len(loader):.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "num_categories": dataset.num_categories, "feature_dim": dataset.feature_dim}, output)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
