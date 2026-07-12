from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SyntheticEvasiveContentDataset
from model import EVADEConfig, EvasiveContentDetector
from multi_agent import multi_agent_predict


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = EvasiveContentDetector(EVADEConfig()).to(device)
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    else:
        print(f"checkpoint not found at {checkpoint_path}; evaluating randomly initialized model")

    dataset = SyntheticEvasiveContentDataset(size=args.test_size, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model.eval()

    full_correct = 0
    partial_correct = 0
    evasion_correct = 0
    multi_agent_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch)
            predictions = outputs["label_logits"].argmax(dim=-1)
            evasion_predictions = (torch.sigmoid(outputs["evasion_logits"]) > 0.5).float()
            multi_agent_predictions = multi_agent_predict(model, batch)

            full_correct += (predictions == batch["label"]).sum().item()
            partial_correct += ((predictions > 0) == (batch["label"] > 0)).sum().item()
            evasion_correct += (evasion_predictions == batch["is_evasive"]).sum().item()
            multi_agent_correct += (multi_agent_predictions == batch["label"]).sum().item()
            total_examples += batch["label"].size(0)

    print({
        "full_accuracy": round(full_correct / total_examples, 4),
        "partial_accuracy": round(partial_correct / total_examples, 4),
        "evasion_accuracy": round(evasion_correct / total_examples, 4),
        "multi_agent_accuracy_untrained_coordinator": round(multi_agent_correct / total_examples, 4),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/evade_toy.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
