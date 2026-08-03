from __future__ import annotations

import argparse

import torch

from data import create_dataloaders
from model import GALA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="gala_toy.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catalog, _, _, test_loader = create_dataloaders(batch_size=args.batch_size)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = GALA(feature_dim=catalog.text_features.size(1), num_items=catalog.text_features.size(0)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    catalog_mm = 0.5 * (checkpoint["catalog_text"] + checkpoint["catalog_image"])
    catalog_mm = catalog_mm.to(device)
    catalog_id = checkpoint["catalog_id"].to(device)

    metrics = {"recall@1": 0.0, "recall@5": 0.0}
    for batch in test_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_metrics = model.evaluate(batch["history"], batch["query"], batch["target"], catalog_mm, catalog_id)
        for key, value in batch_metrics.items():
            metrics[key] += value
    metrics = {key: value / len(test_loader) for key, value in metrics.items()}
    print(metrics)


if __name__ == "__main__":
    main()
