"""
UNIVID Training Script
Paper: arxiv 2606.05748

Training stages:
  1. Caption stage: fine-tune VLM to generate policy-aware captions
  2. Lite stage: train UNIVID-Lite classifier on caption embeddings
  3. Trend stage: fine-tune trend head on new risk categories

Loss functions:
  - Caption: cross-entropy on next-token prediction (standard LM loss)
  - Lite:    binary cross-entropy for multi-label violation detection
             + contrastive loss for embedding quality (from paper Eq. 4)
  - Trend:   cross-entropy on trend category labels
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model import UNIVID, UNIVIDLite
from typing import Dict
from data import VideoModerationDataset, get_dataloader


def contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE / supervised contrastive loss for embedding quality.
    Pulls same-class embeddings together, pushes different-class apart.
    
    From paper Section 3.2: used during lite stage training to ensure
    policy-aligned embeddings cluster by violation type.
    
    Args:
        embeddings: [B, embed_dim] L2-normalized
        labels: [B] integer class labels
        temperature: softmax temperature
    """
    B = embeddings.shape[0]
    embeddings = F.normalize(embeddings, dim=-1)
    sim_matrix = (embeddings @ embeddings.T) / temperature  # [B, B]

    # Mask out self-similarity
    mask_self = torch.eye(B, dtype=torch.bool, device=embeddings.device)
    sim_matrix = sim_matrix.masked_fill(mask_self, float("-inf"))

    # Positive mask: same label
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T) & ~mask_self  # [B, B]

    if pos_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=embeddings.device)

    # For each anchor, compute log-softmax and average over positives
    log_probs = F.log_softmax(sim_matrix, dim=-1)
    loss = -(log_probs * pos_mask.float()).sum(dim=-1) / pos_mask.float().sum(dim=-1).clamp(min=1)
    return loss.mean()


def train_lite_stage(
    univid_model: UNIVID,
    lite_model: UNIVIDLite,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
) -> None:
    """
    Stage 2: Train UNIVID-Lite on cached caption embeddings.
    UNIVID backbone is frozen; only Lite weights are trained.
    """
    optimizer = AdamW(lite_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    bce_loss = nn.BCEWithLogitsLoss()
    best_val_f1 = 0.0

    for epoch in range(args.epochs):
        lite_model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            # Get UNIVID embeddings (frozen)
            all_embeddings = []
            with torch.no_grad():
                for frames, policy in zip(batch["frames"], batch["policy_prompts"]):
                    emb = univid_model.get_caption_embedding(frames, policy)
                    all_embeddings.append(emb)
            embeddings = torch.stack(all_embeddings)  # [B, embed_dim]

            labels = batch["is_violative"].unsqueeze(1)  # [B, 1]
            # Multi-label: one label per class; simplified to binary here
            # In full implementation, use per-class labels [B, num_classes]

            # Forward through Lite
            logits = lite_model(embeddings)  # [B, num_classes]
            binary_labels = labels.expand_as(logits)  # broadcast for demo

            # Classification loss
            cls_loss = bce_loss(logits, binary_labels)

            # Contrastive loss on embeddings
            int_labels = batch["labels"]
            con_loss = contrastive_loss(embeddings, int_labels)

            loss = cls_loss + 0.1 * con_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(lite_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 20 == 0:
                print(f"  [Lite] Epoch {epoch+1}/{args.epochs} "
                      f"Step {step} Loss={loss.item():.4f} "
                      f"(cls={cls_loss.item():.4f}, con={con_loss.item():.4f})")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Validation
        val_metrics = evaluate_lite(univid_model, lite_model, val_loader)
        print(f"  Val F1={val_metrics['f1']:.4f} Precision={val_metrics['precision']:.4f} "
              f"Recall={val_metrics['recall']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(lite_model.state_dict(),
                       os.path.join(args.output_dir, "univid_lite_best.pt"))

    print(f"Training complete. Best Val F1: {best_val_f1:.4f}")


def evaluate_lite(
    univid_model: UNIVID,
    lite_model: UNIVIDLite,
    loader: DataLoader,
    threshold: float = 0.5,
) -> Dict:
    lite_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            embeddings = []
            for frames, policy in zip(batch["frames"], batch["policy_prompts"]):
                emb = univid_model.get_caption_embedding(frames, policy)
                embeddings.append(emb)
            embeddings = torch.stack(embeddings)

            logits = lite_model(embeddings)
            probs = torch.sigmoid(logits[:, 0])  # binary violative/safe
            preds = (probs >= threshold).float()
            all_preds.append(preds.cpu())
            all_labels.append(batch["is_violative"].cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": precision.item(), "recall": recall.item(), "f1": f1.item()}


def main():
    parser = argparse.ArgumentParser(description="Train UNIVID")
    parser.add_argument("--stage", choices=["caption", "lite", "trend"], default="lite")
    parser.add_argument("--data", default="./toy_data")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--toy", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Training UNIVID — stage: {args.stage}")

    if args.toy:
        print("Toy mode: using synthetic data, skipping heavy model loading")
        from data import VideoModerationDataset
        train_ds = VideoModerationDataset(args.data, split="train", toy_size=100)
        val_ds = VideoModerationDataset(args.data, split="val", toy_size=20)
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
        print("In production: load pretrained UNIVID, run caption/lite/trend training stages.")
        print("Toy training loop stub complete.")
        return

    # Full training requires pretrained VLM
    if args.stage == "caption":
        print("Caption stage: fine-tune VLM on policy-aware captions")
        print("  Loss: next-token cross-entropy on caption text")
        print("  Data: expert-annotated violative + clean videos with policy captions")

    elif args.stage == "lite":
        print("Lite stage: train UNIVID-Lite classifier on caption embeddings")
        print("  Loss: BCE + contrastive")
        print("  Frozen: UNIVID backbone")

    elif args.stage == "trend":
        print("Trend stage: fine-tune trend head for new risk categories")
        print("  Loss: cross-entropy on trend labels")
        print("  Frozen: everything except trend_head")


if __name__ == "__main__":
    main()
