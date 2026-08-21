from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from dataset import make_loader
from model import PolicyGuideScorer


def train(epochs: int, output_dir: Path) -> None:
    loader, vocab, node_to_id, id_to_node = make_loader(batch_size=4, shuffle=True)
    model = PolicyGuideScorer(len(vocab), len(node_to_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for token_ids, offsets, current_nodes, targets in loader:
            logits = model(token_ids, offsets, current_nodes)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.numel()
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            total += targets.numel()
        print(f"epoch={epoch} loss={total_loss / total:.4f} acc={correct / total:.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": vocab, "node_to_id": node_to_id, "id_to_node": id_to_node}, output_dir / "policyguide.pt")
    (output_dir / "metadata.json").write_text(json.dumps({"epochs": epochs, "vocab_size": len(vocab), "nodes": node_to_id}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    train(args.epochs, args.output_dir)
