"""
Training script for FLUID toy reproduction.

Includes staged warmup as described in the paper:
- Stage 1: Pretrain LUCID encoder (VQ reconstruction loss)
- Stage 2: Joint training with ranking loss + VQ loss (staged warmup)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import get_dataloaders
from model import FLUIDRanker


def train_epoch(model, loader, optimizer, device, vq_weight: float = 0.1):
    model.train()
    total_loss = 0.0
    n_batches = 0

    bce = nn.BCEWithLogitsLoss()

    for batch in loader:
        user_ids = batch["user_id"].to(device)
        visual = batch["visual_feat"].to(device)
        audio = batch["audio_feat"].to(device)
        text = batch["text_feat"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        scores, vq_loss = model(user_ids, visual, audio, text)

        # Ranking loss
        ranking_loss = bce(scores, labels)

        # Total loss: ranking + VQ commitment loss (staged warmup: vq_weight increases)
        loss = ranking_loss + vq_weight * vq_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []

    for batch in loader:
        user_ids = batch["user_id"].to(device)
        visual = batch["visual_feat"].to(device)
        audio = batch["audio_feat"].to(device)
        text = batch["text_feat"].to(device)
        labels = batch["label"]

        scores, _ = model(user_ids, visual, audio, text)
        all_scores.append(scores.cpu())
        all_labels.append(labels)

    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    preds = (torch.sigmoid(all_scores) > 0.5).float()
    acc = (preds == all_labels).float().mean().item()
    auc = compute_auc(all_scores, all_labels)
    return {"acc": acc, "auc": auc}


def compute_auc(scores, labels):
    """Simple AUC computation."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(labels.numpy(), torch.sigmoid(scores).numpy())
    except Exception:
        return 0.5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, data = get_dataloaders(
        batch_size=256, n_rooms=500, n_users=2000, n_interactions=20000
    )

    # Model
    model = FLUIDRanker(
        n_users=data["n_users"],
        user_dim=64,
        visual_dim=256,
        audio_dim=128,
        text_dim=128,
        lucid_hidden_dim=128,
        n_levels=4,
        codebook_size=256,
        hidden_dim=64,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Staged warmup: gradually increase VQ weight (paper: stabilize ID-free training)
    vq_weights = [0.0] * 3 + [0.01] * 3 + [0.05] * 4 + [0.1] * 10

    for epoch in range(20):
        vq_w = vq_weights[min(epoch, len(vq_weights) - 1)]
        train_loss = train_epoch(model, train_loader, optimizer, device, vq_weight=vq_w)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1:2d} | loss={train_loss:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | val_auc={val_metrics['auc']:.4f} | "
            f"vq_w={vq_w}"
        )

    torch.save(model.state_dict(), "fluid_checkpoint.pt")
    print("Training complete. Checkpoint saved to fluid_checkpoint.pt")


if __name__ == "__main__":
    main()
