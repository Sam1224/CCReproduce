"""
GLiGuard — Training script
Schema-conditioned multi-aspect safety classifier training.

Usage:
    python train.py --epochs 5 --lr 2e-4 --batch_size 16
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import LLMSafetyDataset, collate_fn, ALL_SCHEMA_DIMS
from model import GLiGuard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_loss(
    model: GLiGuard,
    prompts: List[str],
    responses: List[str],
    active_schemas: List[List[str]],
    labels: List[Dict[str, int]],
    device: str,
) -> torch.Tensor:
    """
    Compute multi-task BCE loss over all active schema dimensions.

    Paper: parallel multi-task classification — all dimensions in one forward pass.
    Loss = mean of per-dimension binary cross-entropy losses.
    """
    dim_logits = model(prompts, responses, active_schemas)

    if not dim_logits:
        return torch.tensor(0.0, requires_grad=True, device=device)

    criterion = nn.CrossEntropyLoss()
    losses = []

    # Build per-dim label tensors from sample labels
    dim_labels: Dict[str, List[int]] = {}
    for b_idx, (schema, sample_labels) in enumerate(zip(active_schemas, labels)):
        for dim in schema:
            if dim in sample_labels:
                if dim not in dim_labels:
                    dim_labels[dim] = []
                dim_labels[dim].append(sample_labels[dim])

    for dim, logits in dim_logits.items():
        if dim not in dim_labels:
            continue
        label_tensor = torch.tensor(dim_labels[dim], dtype=torch.long, device=device)
        # Align lengths (logits may differ if some samples don't have this dim)
        min_len = min(len(logits), len(label_tensor))
        if min_len == 0:
            continue
        loss = criterion(logits[:min_len], label_tensor[:min_len])
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, requires_grad=True, device=device)

    return torch.stack(losses).mean()


def train(args):
    device = args.device

    # Data
    train_ds = LLMSafetyDataset(split="train", num_samples=args.num_samples)
    val_ds = LLMSafetyDataset(split="val", num_samples=args.num_samples // 5, seed=999)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn
    )

    # Model
    model = GLiGuard(
        schema_dims=ALL_SCHEMA_DIMS,
        hidden_dim=256,
        num_encoder_layers=2,
        device=device,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"GLiGuard | {n_params:.1f}M parameters")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    best_val_loss = float("inf")
    Path(args.save_dir).mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch in train_loader:
            loss = compute_loss(
                model,
                batch["prompts"], batch["responses"],
                batch["active_schemas"], batch["labels"],
                device,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                loss = compute_loss(
                    model,
                    batch["prompts"], batch["responses"],
                    batch["active_schemas"], batch["labels"],
                    device,
                )
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = Path(args.save_dir) / "gliguard_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  ✓ Best checkpoint saved → {ckpt_path}")

    logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="GLiGuard training")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save_dir", default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
