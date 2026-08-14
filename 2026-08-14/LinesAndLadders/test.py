from __future__ import annotations

import argparse

import torch

from data import create_dataloaders
from model import LinesAndLadders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="lines_ladders_toy.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = create_dataloaders(batch_size=args.batch_size, seed=args.seed)
    model = LinesAndLadders().to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    metrics = {
        "line_precision": 0.0,
        "line_recall": 0.0,
        "line_f1": 0.0,
        "ladder_precision": 0.0,
        "ladder_recall": 0.0,
        "ladder_f1": 0.0,
    }
    for batch in test_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_metrics = model.evaluate(batch)
        for key, value in batch_metrics.items():
            metrics[key] += value
    metrics = {key: value / len(test_loader) for key, value in metrics.items()}
    print(metrics)


if __name__ == "__main__":
    main()
