from __future__ import annotations

import argparse

import torch

from data import create_dataloaders
from model import TimeRoute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="timeroute_toy.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catalog, _, test_loader = create_dataloaders(batch_size=args.batch_size, seed=args.seed)
    model = TimeRoute(num_items=catalog.text_embeddings.size(0), max_length=test_loader.dataset.sessions.size(1)).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"])
    catalog_tensors = {key: value.to(device) for key, value in payload["catalog"].items()}
    model.eval()

    metrics = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "ndcg@10": 0.0}
    for batch in test_loader:
        session = batch["session"].to(device)
        timestamps = batch["timestamps"].to(device)
        target = batch["target"].to(device)
        batch_metrics = model.evaluate(session, timestamps, target, catalog_tensors)
        for key, value in batch_metrics.items():
            metrics[key] += value
    metrics = {key: value / len(test_loader) for key, value in metrics.items()}
    print(metrics)


if __name__ == "__main__":
    main()
