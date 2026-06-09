"""
UNIVID — Training Script

Two-phase training (faithful to paper §4.2):
  Phase 1: SFT on general video-captioning data
  Phase 2: Instruction-tune on policy-specific moderation data
           (hybrid: expert annotations + high-quality synthetic data)
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, CLIPImageProcessor
from model import UNIVID, ModerationClassificationHead
from dataset import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/toy_videos")
    parser.add_argument("--llm_name", default="Qwen/Qwen2-1.5B")
    parser.add_argument("--output_dir", default="checkpoints/univid")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_violation_types", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_phase1_captioning(model, dataloader, optimizer, scheduler, device, epoch):
    """Phase 1: SFT on caption generation objective."""
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        policy_input_ids = batch["policy_input_ids"].to(device)
        caption_input_ids = batch["caption_input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            policy_input_ids=policy_input_ids,
            caption_input_ids=caption_input_ids,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        if step % 10 == 0:
            print(f"  [Phase1] Epoch {epoch} Step {step} Loss: {loss.item():.4f}")

    scheduler.step()
    return total_loss / len(dataloader)


def train_phase2_moderation(
    model, cls_head, dataloader, optimizer, scheduler, device, epoch
):
    """
    Phase 2: Joint training on:
      - Caption generation loss (L_cap)
      - Violation classification loss (L_cls)

    Combined objective:
        L = λ * L_cap + (1 - λ) * L_cls
    """
    model.train()
    cls_head.train()
    criterion_cls = nn.BCEWithLogitsLoss()
    lambda_cap = 0.7  # balance caption and classification
    total_loss = 0

    for step, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        policy_input_ids = batch["policy_input_ids"].to(device)
        caption_input_ids = batch["caption_input_ids"].to(device)
        labels = batch["labels"].to(device)
        violation_labels = batch["violation_labels"].to(device)

        # Caption generation loss
        cap_outputs = model(
            pixel_values=pixel_values,
            policy_input_ids=policy_input_ids,
            caption_input_ids=caption_input_ids,
            labels=labels,
        )
        l_cap = cap_outputs.loss

        # Get caption embeddings for classification
        emb = model.get_caption_embedding(pixel_values, policy_input_ids)
        logits = cls_head(emb)
        l_cls = criterion_cls(logits, violation_labels)

        loss = lambda_cap * l_cap + (1 - lambda_cap) * l_cls
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(cls_head.parameters()), 1.0
        )
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        if step % 10 == 0:
            print(
                f"  [Phase2] Epoch {epoch} Step {step} "
                f"L_cap={l_cap.item():.4f} L_cls={l_cls.item():.4f}"
            )

    scheduler.step()
    return total_loss / len(dataloader)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch16"
    )

    model = UNIVID(llm_name=args.llm_name).to(device)
    cls_head = ModerationClassificationHead(
        hidden_dim=model.llm.config.hidden_size,
        num_violation_types=args.num_violation_types,
    ).to(device)

    train_loader = build_dataloader(
        args.data_path, tokenizer, image_processor, "train", args.batch_size
    )

    # Phase 1: Caption SFT
    print("=== Phase 1: Caption SFT ===")
    opt1 = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch1 = CosineAnnealingLR(opt1, T_max=args.epochs)
    for epoch in range(1, args.epochs + 1):
        loss = train_phase1_captioning(model, train_loader, opt1, sch1, device, epoch)
        print(f"Phase1 Epoch {epoch} AvgLoss={loss:.4f}")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "phase1.pt"))

    # Phase 2: Policy-specific moderation fine-tune
    print("\n=== Phase 2: Moderation Instruction-Tuning ===")
    opt2 = AdamW(
        list(model.parameters()) + list(cls_head.parameters()),
        lr=args.lr * 0.5,
        weight_decay=0.01,
    )
    sch2 = CosineAnnealingLR(opt2, T_max=args.epochs)
    for epoch in range(1, args.epochs + 1):
        loss = train_phase2_moderation(
            model, cls_head, train_loader, opt2, sch2, device, epoch
        )
        print(f"Phase2 Epoch {epoch} AvgLoss={loss:.4f}")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "univid_final.pt"))
    torch.save(cls_head.state_dict(), os.path.join(args.output_dir, "cls_head.pt"))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
