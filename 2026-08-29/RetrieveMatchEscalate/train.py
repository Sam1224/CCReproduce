from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset import build_datasets, collate_product_pairs
from model import RetrieveMatchEscalate


def train_epoch(model: RetrieveMatchEscalate, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {"loss": 0.0, "retrieve_loss": 0.0, "match_loss": 0.0, "escalation_rate": 0.0}
    batch_count = 0
    for batch in loader:
        tensor_batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        loss, metrics = model.loss(tensor_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for metric_name, metric_value in metrics.items():
            totals[metric_name] += metric_value
        batch_count += 1
    return {metric_name: metric_value / max(1, batch_count) for metric_name, metric_value in totals.items()}


def evaluate(model: RetrieveMatchEscalate, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total = 0
    escalation_count = 0
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            output = model(tensor_batch)
            predictions = (output.final_probability >= 0.5).float()
            labels = tensor_batch["label"]
            true_positive += int(((predictions == 1) & (labels == 1)).sum().cpu())
            false_positive += int(((predictions == 1) & (labels == 0)).sum().cpu())
            false_negative += int(((predictions == 0) & (labels == 1)).sum().cpu())
            total += int(labels.numel())
            escalation_count += int(output.escalate_mask.sum().cpu())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1_score = 2 * precision * recall / max(1e-8, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1_score, "escalation_rate": escalation_count / max(1, total)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a toy Retrieve-Match-Escalate product linking model.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rme_toy"))
    args = parser.parse_args()

    train_dataset, test_dataset, vocab = build_datasets()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_product_pairs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_product_pairs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetrieveMatchEscalate(vocab_size=len(vocab)).to(device)
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
    print(f"Saved checkpoint to {args.output_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
