from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import build_datasets, collate_examples
from model import DMEModel
from train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DME toy checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/dme_toy/checkpoint.pt"))
    args = parser.parse_args()
    _, test_dataset, vocab = build_datasets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMEModel(vocab_size=len(vocab)).to(device)
    if args.checkpoint.exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state"])
    loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_examples)
    print(evaluate(model, loader, device))


if __name__ == "__main__":
    main()
