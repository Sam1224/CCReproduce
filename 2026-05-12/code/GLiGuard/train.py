"""
GLiGuard training script.

Training procedure (Section 4 in paper):
  - Fine-tune the bidirectional encoder + classification heads jointly
  - Loss: multi-label BCE summed over all active label slots per example
  - Optimizer: AdamW with linear warmup + cosine decay
  - Each batch contains examples from all task types (multi-task co-training)

Usage:
    python data.py --output toy_data.jsonl
    python train.py --data toy_data.jsonl --epochs 3 --output ./checkpoints
"""

import json
import argparse
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from model import GLiGuardModel, GLiGuardLoss, build_model
from schemas import (
    ALL_SCHEMAS,
    PROMPT_SAFETY_SCHEMA,
    RESPONSE_SAFETY_SCHEMA,
    REFUSAL_SCHEMA,
    HARM_CATEGORY_SCHEMA,
    JAILBREAK_SCHEMA,
    get_all_label_names,
)

SCHEMA_MAP = {s["task_name"]: s for s in ALL_SCHEMAS}


class GLiGuardDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Return batch as list of dicts (variable-length content strings)."""
    return batch


def train_epoch(
    model: GLiGuardModel,
    loader: DataLoader,
    optimizer,
    scheduler,
    criterion: GLiGuardLoss,
    device: torch.device,
    log_every: int = 20,
) -> float:
    model.train()
    total_loss = 0.0
    n_steps = 0

    for step, batch in enumerate(loader):
        optimizer.zero_grad()

        batch_loss = torch.tensor(0.0, device=device, requires_grad=False)
        valid_examples = 0

        for example in batch:
            task_name = example["task"]
            if task_name not in SCHEMA_MAP:
                continue

            schema = SCHEMA_MAP[task_name]
            content = example["content"]
            label_dict = example["labels"]

            # Forward pass (single example)
            logits = model.forward(content, [schema], device)

            # Build target tensors
            targets: dict[str, dict[str, torch.Tensor]] = {}
            targets[task_name] = {
                label: torch.tensor(float(label_dict.get(label, 0)), device=device)
                for label in get_all_label_names(schema)
            }

            loss = criterion(logits, targets)
            batch_loss = batch_loss + loss
            valid_examples += 1

        if valid_examples > 0:
            # Average loss over examples in the batch
            avg_loss = batch_loss / valid_examples
            avg_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += avg_loss.item()
            n_steps += 1

        if (step + 1) % log_every == 0:
            print(f"  step {step+1}/{len(loader)} | loss {total_loss/n_steps:.4f}")

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def evaluate(
    model: GLiGuardModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Compute per-task macro F1 and overall accuracy."""
    model.eval()

    from collections import defaultdict
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    correct = 0
    total = 0

    for batch in loader:
        for example in batch:
            task_name = example["task"]
            if task_name not in SCHEMA_MAP:
                continue
            schema = SCHEMA_MAP[task_name]
            preds = model.predict(example["content"], [schema], device=device)
            gold = example["labels"]

            for label in get_all_label_names(schema):
                p = preds.get(task_name, {}).get(label, 0)
                g = int(gold.get(label, 0))
                key = f"{task_name}/{label}"
                if p == 1 and g == 1:
                    tp[key] += 1
                elif p == 1 and g == 0:
                    fp[key] += 1
                elif p == 0 and g == 1:
                    fn[key] += 1

                correct += int(p == g)
                total += 1

    # Per-label F1
    f1_scores = {}
    for key in set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())):
        precision = tp[key] / max(tp[key] + fp[key], 1)
        recall = tp[key] / max(tp[key] + fn[key], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        f1_scores[key] = f1

    macro_f1 = sum(f1_scores.values()) / max(len(f1_scores), 1)
    accuracy = correct / max(total, 1)

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "per_label_f1": f1_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Train GLiGuard")
    parser.add_argument("--data", default="toy_data.jsonl", help="JSONL dataset")
    parser.add_argument("--encoder", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--output", default="./checkpoints", help="Checkpoint dir")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Dataset not found at {args.data}. Run: python data.py --output {args.data}")
        return

    with open(data_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    random.shuffle(data)
    split = int(len(data) * (1 - args.val_split))
    train_data, val_data = data[:split], data[split:]
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    train_loader = DataLoader(
        GLiGuardDataset(train_data), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        GLiGuardDataset(val_data), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn
    )

    # Build model
    model = build_model(args.encoder).to(device)
    print(f"Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Total trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    criterion = GLiGuardLoss()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        print(f"Train loss: {train_loss:.4f}")

        metrics = evaluate(model, val_loader, device)
        print(f"Val macro F1: {metrics['macro_f1']:.4f} | Accuracy: {metrics['accuracy']:.4f}")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            ckpt_path = out_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": model.config,
                "val_macro_f1": best_f1,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (F1={best_f1:.4f}) → {ckpt_path}")

    print(f"\nTraining complete. Best Val Macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
