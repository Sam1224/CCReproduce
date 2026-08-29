from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import build_datasets, collate_product_pairs
from model import RetrieveMatchEscalate
from train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Retrieve-Match-Escalate toy checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/rme_toy/checkpoint.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    _, test_dataset, vocab = build_datasets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetrieveMatchEscalate(vocab_size=len(vocab)).to(device)
    if args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        print(f"Checkpoint {args.checkpoint} not found; evaluating randomly initialized model.")
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_product_pairs)
    metrics = evaluate(model, loader, device)
    print(metrics)


if __name__ == "__main__":
    main()
