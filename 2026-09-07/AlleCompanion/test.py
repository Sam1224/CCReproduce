from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, random_split

from dataset import ToyComplementDataset
from model import TwoTower


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/allecompanion.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ToyComplementDataset()
    _, test_set = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))], generator=torch.Generator().manual_seed(5))
    model = TwoTower(feature_dim=dataset.feature_dim, num_categories=dataset.num_categories).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for batch in DataLoader(test_set, batch_size=args.batch_size):
            batch = {key: value.to(device) for key, value in batch.items()}
            scores = torch.sigmoid(model(batch))
            pred = (scores >= 0.5).float()
            correct += int((pred == batch["label"]).sum().item())
            total += pred.numel()
    print({"pair_accuracy": round(correct / max(total, 1), 4)})


if __name__ == "__main__":
    main()
