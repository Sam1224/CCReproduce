import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import make_splits
from model import TaoLiveHATModel


def evaluate(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x, y)
            preds = out.logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()
            loss_sum += out.loss.item() * y.size(0)
    return {"accuracy": correct / max(total, 1), "loss": loss_sum / max(total, 1)}


def train_one(name, dataset, val_dataset, out_dir: Path, device: torch.device):
    model = TaoLiveHATModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    best = {"accuracy": -1.0}
    best_path = out_dir / f"{name}.pt"

    for epoch in range(18):
        model.train()
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x, y)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()

        metrics = evaluate(model, val_dataset, device)
        if metrics["accuracy"] > best["accuracy"]:
            best = {"epoch": epoch + 1, **metrics}
            torch.save(model.state_dict(), best_path)

    return {"checkpoint": str(best_path.name), **best}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = make_splits()
    summary = {
        "fixed_harness": train_one("fixed_harness", splits["fixed_train"], splits["val"], out_dir, device),
        "hat": train_one("hat", splits["hat_train"], splits["val"], out_dir, device),
    }

    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
