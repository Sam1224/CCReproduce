from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SyntheticEvasiveContentDataset
from model import EVADEConfig, EvasiveContentDetector, compute_loss


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = SyntheticEvasiveContentDataset(size=args.train_size, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = EvasiveContentDetector(EVADEConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss = compute_loss(outputs, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch["label"].size(0)
            total_correct += (outputs["label_logits"].argmax(dim=-1) == batch["label"]).sum().item()
            total_examples += batch["label"].size(0)
        print(f"epoch={epoch} loss={total_loss / total_examples:.4f} full_acc={total_correct / total_examples:.4f}")

    checkpoint_path = Path(args.output)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": EVADEConfig().__dict__}, checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=str, default="checkpoints/evade_toy.pt")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
