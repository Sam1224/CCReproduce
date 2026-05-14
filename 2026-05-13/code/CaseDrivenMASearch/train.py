"""
GRM training script (standalone — without agents).

Usage:
    python train.py --data data/toy_ecom_pairs.jsonl --epochs 10 --lr 1e-3
"""

import argparse
import json
import random
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.generative_relevance_model import GenerativeRelevanceModel, LABEL_SPACE


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def train(data_path, epochs, lr, output_path="outputs/grm.pt"):
    import os; os.makedirs("outputs", exist_ok=True)

    pairs = load_jsonl(data_path)
    train_set = [p for p in pairs if p.get("split") == "train"]
    test_set  = [p for p in pairs if p.get("split") == "test"]
    label2idx = {l: i for i, l in enumerate(LABEL_SPACE)}

    model = GenerativeRelevanceModel()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training GRM: {len(train_set)} train / {len(test_set)} test | "
          f"epochs={epochs} | lr={lr}")

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_set)
        total_loss = 0.0
        for pair in train_set:
            gold = pair.get("label")
            if gold not in label2idx:
                continue
            label_t = torch.tensor([label2idx[gold]])
            out = model.forward(
                [pair["query"]],
                [pair["product_title"]],
                [pair["product_desc"]],
                labels=label_t,
            )
            optimizer.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += out.loss.item()
        scheduler.step()

        # ── Eval ──────────────────────────────────────────────────────────────
        model.eval()
        correct = 0
        with torch.no_grad():
            for pair in test_set:
                pred, _ = model.predict(
                    pair["query"], pair["product_title"], pair["product_desc"]
                )
                if pred == pair.get("label"):
                    correct += 1
        acc = correct / max(len(test_set), 1)
        avg_loss = total_loss / max(len(train_set), 1)
        print(f"Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | test_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), output_path)

    print(f"\nBest test accuracy: {best_acc:.4f}")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/toy_ecom_pairs.jsonl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="outputs/grm.pt")
    args = parser.parse_args()
    train(args.data, args.epochs, args.lr, args.output)
