"""
GLiGuard Training Script
Reproduction of arXiv:2605.07982

Training strategy (from paper):
  - Fine-tune a ~0.3B bidirectional encoder (DeBERTa-v3-large or GLiNER2 backbone)
  - Multi-label BCE loss applied only over schema-active tasks (masked BCE)
  - AdamW optimizer with linear warmup + cosine decay
  - Data from WildGuard, PromptBench, and curated safety datasets
"""

import os
import argparse
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm

from model import GLiGuard, ALL_LABELS, LABEL_TO_IDX
from data import GLiGuardDataset, generate_toy_samples, load_wildguard_format


# ---------------------------------------------------------------------------
# Schema-aware masked BCE loss
# Paper: loss is computed only over schema-specified (active) tasks
# ---------------------------------------------------------------------------

class MaskedBCELoss(nn.Module):
    """
    Binary cross-entropy loss applied only to active (schema-specified) tasks.

    This prevents the model from being penalized for tasks not in the schema,
    which could introduce conflicting gradients across different schema configs.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,        # [batch, num_labels]
        labels: torch.Tensor,        # [batch, num_labels]
        active_mask: torch.Tensor,   # [batch, num_labels] bool
    ) -> torch.Tensor:
        # Per-element BCE loss
        loss_per_element = self.bce(logits, labels)    # [batch, num_labels]
        # Mask: only active tasks contribute to gradient
        masked_loss = loss_per_element * active_mask.float()
        # Mean over active task-sample pairs
        num_active = active_mask.float().sum().clamp(min=1)
        return masked_loss.sum() / num_active


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = output["probs"].cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds, axis=0)    # [N, num_labels]
    all_labels = np.concatenate(all_labels, axis=0)  # [N, num_labels]

    # Macro F1 over all labels
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    # Per-task F1 for the three core safety tasks
    per_task_f1 = {}
    for task in ["prompt_safety", "response_safety", "refusal_detection"]:
        idx = LABEL_TO_IDX[task]
        per_task_f1[task] = f1_score(
            all_labels[:, idx], all_preds[:, idx], zero_division=0
        )

    return {"macro_f1": macro_f1, **per_task_f1}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)

    if args.data_dir and os.path.exists(os.path.join(args.data_dir, "train.jsonl")):
        print("Loading WildGuard-format data...")
        from data import load_wildguard_format
        samples = load_wildguard_format(os.path.join(args.data_dir, "train.jsonl"))
    else:
        print("No data_dir found — using toy dataset for interface verification...")
        samples = generate_toy_samples(n=400)

    train_size = int(0.9 * len(samples))
    val_size = len(samples) - train_size
    train_samples, val_samples = random_split(samples, [train_size, val_size])

    train_dataset = GLiGuardDataset(list(train_samples), tokenizer, args.max_length)
    val_dataset = GLiGuardDataset(list(val_samples), tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # --- Model ---
    model = GLiGuard(encoder_name=args.encoder_name).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # --- Optimizer & scheduler ---
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    criterion = MaskedBCELoss()

    # --- Training ---
    best_val_f1 = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            active_mask = batch["active_mask"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(output["logits"], labels, active_mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"  step {step+1}/{len(train_loader)}  loss={avg_loss:.4f}")

        # --- Validation ---
        metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch} | macro_F1={metrics['macro_f1']:.4f} | "
            f"prompt_safety_F1={metrics['prompt_safety']:.4f} | "
            f"response_safety_F1={metrics['response_safety']:.4f}"
        )

        if metrics["macro_f1"] > best_val_f1:
            best_val_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pt"))
            print(f"  ✓ New best model saved (macro_F1={best_val_f1:.4f})")

    print(f"\nTraining complete. Best macro_F1: {best_val_f1:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train GLiGuard")
    parser.add_argument("--encoder_name", default="microsoft/deberta-v3-large",
                        help="Pretrained encoder (DeBERTa-v3-large ≈ 0.3B params)")
    parser.add_argument("--data_dir", default="./data",
                        help="Directory containing train.jsonl (WildGuard format)")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
