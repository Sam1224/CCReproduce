import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import make_splits
from model import SMEOPipeline


def eval_utility(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    loss_sum = 0.0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model.forward_utility(x, y)
            loss_sum += out.loss.item() * x.size(0)
            total += x.size(0)
    return loss_sum / max(total, 1)


def eval_rank(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model.forward_rank(x, y)
            correct += (out.logits.argmax(dim=-1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = make_splits()
    model = SMEOPipeline().to(device)
    utility_optim = torch.optim.AdamW(model.utility.parameters(), lr=2e-3)
    rank_optim = torch.optim.AdamW(model.ranker.parameters(), lr=2e-3)

    utility_loader = DataLoader(splits["utility_train"], batch_size=64, shuffle=True)
    rank_loader = DataLoader(splits["rank_train"], batch_size=64, shuffle=True)

    best_utility = {"loss": 1e9}
    best_rank = {"acc": -1.0}

    for epoch in range(20):
        model.train()
        for batch in utility_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model.forward_utility(x, y)
            utility_optim.zero_grad()
            out.loss.backward()
            utility_optim.step()

        val_loss = eval_utility(model, splits["utility_val"], device)
        if val_loss < best_utility["loss"]:
            best_utility = {"epoch": epoch + 1, "loss": val_loss}
            torch.save(model.utility.state_dict(), out_dir / "utility.pt")

    for epoch in range(16):
        model.train()
        for batch in rank_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model.forward_rank(x, y)
            rank_optim.zero_grad()
            out.loss.backward()
            rank_optim.step()

        val_acc = eval_rank(model, splits["rank_val"], device)
        if val_acc > best_rank["acc"]:
            best_rank = {"epoch": epoch + 1, "acc": val_acc}
            torch.save(model.ranker.state_dict(), out_dir / "ranker.pt")

    summary = {"utility": best_utility, "ranker": best_rank}
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
