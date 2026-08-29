from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset import build_datasets, collate_examples
from model import DMEModel


def train_epoch(model: DMEModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "accuracy": 0.0}
    batches = 0
    for batch in loader:
        tensor_batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        loss, metrics = model.loss(tensor_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for metric_name, metric_value in metrics.items():
            totals[metric_name] += metric_value
        batches += 1
    return {metric_name: metric_value / max(1, batches) for metric_name, metric_value in totals.items()}


def evaluate(model: DMEModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            logits = model(tensor_batch)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((predictions == tensor_batch["label"]).sum().cpu())
            total += int(predictions.numel())
    return {"accuracy": correct / max(1, total)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a toy Douyin Multimodal Embedding model.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dme_toy"))
    args = parser.parse_args()

    train_dataset, test_dataset, vocab = build_datasets()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_examples)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_examples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMEModel(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        eval_metrics = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "eval": eval_metrics})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(history[-1], ensure_ascii=False, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "vocab": vocab.token_to_id, "history": history}, args.output_dir / "checkpoint.pt")
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(history[-1], metrics_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
