"""
UNIVID Training Script
Two-stage training:
    Stage 1: SFT on (video, policy) → caption pairs
    Stage 2: Moderation fine-tuning with violation labels

Usage:
    python train.py --stage 1 --epochs 3 --batch_size 8
    python train.py --stage 2 --epochs 5 --batch_size 8
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.univid import UNIVID
from pipeline.moderation_actor import ModerationActor
from data.dataset import ModerationDataset


def train_stage1_sft(model: UNIVID, dataloader: DataLoader, epochs: int, device: str):
    """Stage 1: Supervised fine-tuning — caption generation."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            frames = batch["frames"].to(device)              # (B, T, 3, H, W)
            policy_ids = batch["policy_ids"].to(device)      # (B, P)
            caption_ids = batch["caption_ids"].to(device)    # (B, L)

            optimizer.zero_grad()
            outputs = model(frames, policy_ids, caption_ids)
            logits = outputs["caption_logits"]  # (B, L, vocab)

            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = caption_ids[:, 1:].contiguous().view(-1)
            loss = criterion(shift_logits, shift_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Stage 1] Epoch {epoch+1}/{epochs} | Caption SFT Loss: {avg_loss:.4f}")


def train_stage2_moderation(
    backbone: UNIVID,
    actor: ModerationActor,
    dataloader: DataLoader,
    epochs: int,
    device: str,
):
    """Stage 2: Fine-tune moderation heads with violation labels."""
    # Freeze backbone except LoRA adapters
    for name, param in backbone.named_parameters():
        param.requires_grad = "lora" in name

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(actor.parameters()),
        lr=1e-4,
        weight_decay=0.01,
    )
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            frames = batch["frames"].to(device)
            policy_ids = batch["policy_ids"].to(device)
            labels = batch["labels"].to(device)  # (B, C) multi-label

            optimizer.zero_grad()
            outputs = backbone(frames, policy_ids)
            caption_emb = outputs["caption_embedding"]  # (B, embed_dim)

            actor_out = actor(caption_emb)
            pred = actor_out["combined_scores"]  # (B, C)

            # Pad/trim labels to match num_categories (64 in model, 10 in toy dataset)
            C = pred.shape[1]
            if labels.shape[1] < C:
                pad = torch.zeros(labels.shape[0], C - labels.shape[1], device=device)
                labels = torch.cat([labels, pad], dim=1)
            elif labels.shape[1] > C:
                labels = labels[:, :C]

            loss = criterion(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(actor.parameters()), 1.0
            )
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Stage 2] Epoch {epoch+1}/{epochs} | Moderation BCE Loss: {avg_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device
    dataset = ModerationDataset(size=args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    backbone = UNIVID().to(device)
    actor = ModerationActor(embed_dim=256, num_categories=64).to(device)

    if args.stage == 1:
        print("=== Stage 1: Caption SFT ===")
        train_stage1_sft(backbone, dataloader, args.epochs, device)
    else:
        print("=== Stage 2: Moderation Fine-tuning ===")
        train_stage2_moderation(backbone, actor, dataloader, args.epochs, device)

    print("Training complete.")


if __name__ == "__main__":
    main()
