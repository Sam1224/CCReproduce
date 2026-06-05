"""
Train the OCL PolicyMLP on synthetic negotiation action data.

Usage:
    python data.py --n_samples 5000 --out data/actions.jsonl
    python train.py --data data/actions.jsonl --epochs 20 --ckpt checkpoints/policy.pt
"""

import json
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from policy import PolicyMLP, FEATURE_DIM


class ActionDataset(Dataset):
    def __init__(self, path: str):
        self.records = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                self.records.append((r["features"], r["label"]))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        feats, label = self.records[idx]
        return (torch.tensor(feats, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))


def train(data_path: str, epochs: int, ckpt_path: str, lr: float = 1e-3,
          batch_size: int = 128, val_split: float = 0.2, seed: int = 42):

    torch.manual_seed(seed)
    dataset = ActionDataset(data_path)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = PolicyMLP(input_dim=FEATURE_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x)
                val_loss += criterion(pred, y).item() * len(x)
                correct += ((pred > 0.5) == y.bool()).sum().item()
        val_loss /= n_val
        val_acc = correct / n_val

        print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved checkpoint → {ckpt_path}")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/actions.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ckpt", default="checkpoints/policy.pt")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    train(args.data, args.epochs, args.ckpt, args.lr, args.batch_size)
