from __future__ import annotations

import argparse

import torch

from data import PRIMITIVES, create_dataloaders
from model import EvoReason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="evoreason_toy.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catalog, _, test_loader = create_dataloaders(batch_size=args.batch_size)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = EvoReason(
        num_items=catalog.item_features.size(0),
        num_primitives=len(PRIMITIVES),
        feature_dim=catalog.item_features.size(1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    item_features = checkpoint["item_features"].to(device)

    metrics = {"recall@1": 0.0, "recall@5": 0.0, "primitive_acc": 0.0}
    for batch in test_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_metrics = model.evaluate(batch["history"], batch["target"], batch["primitive_labels"], item_features)
        for key, value in batch_metrics.items():
            metrics[key] += value
    metrics = {key: value / len(test_loader) for key, value in metrics.items()}
    print(metrics)


if __name__ == "__main__":
    main()
