"""
Training script for UNIVID reproduction.

Paper: UNIVID: Unified Vision-Language Model for Video Moderation (arXiv:2606.05748)
Authors: Kejuan Yang et al., ByteDance

Training strategy (from paper):
  - Phase 1: Train UNIVID backbone jointly on caption generation + violation prediction
  - Phase 2: Freeze backbone; train UNIVID-Lite on pooled embeddings
  - Phase 3: Build RAG index from violation examples in training set
  - Phase 4: Train Trend Governance Head on emerging violation clusters

Loss:   L = λ_cap * L_cap + λ_vio * L_vio + λ_pol * L_pol  [Eq. 3 simplified]
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path

from data import generate_synthetic_dataset, ModerationDataset, POLICY_CATEGORIES
from model import UNIVIDSystem


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNIVID for video moderation")
    parser.add_argument("--lm_model", default="distilgpt2", help="Base LM model name")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs_backbone", type=int, default=3)
    parser.add_argument("--epochs_lite", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--visual_dim", type=int, default=768)
    parser.add_argument("--max_caption_len", type=int, default=64)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--violation_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor):
    """Binary classification metrics."""
    preds = (torch.sigmoid(logits) > 0.5).long()
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    tn = ((preds == 0) & (labels == 0)).sum().float()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-9)

    # Violation leakage = FN / (TP + FN) = 1 - recall (false negative rate)
    leakage = fn / (tp + fn + 1e-9)
    # Overkill = FP / (FP + TN) (false positive rate)
    overkill = fp / (fp + tn + 1e-9)

    return {
        "accuracy": acc.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "violation_leakage": leakage.item(),
        "overkill_rate": overkill.item(),
    }


def train_epoch(model, loader, optimizer, device, stage="backbone"):
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc=f"Train [{stage}]", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        visual_features = batch["visual_features"].to(device)
        policy_label = batch["policy_label"].to(device)
        violation_label = batch["violation_label"].to(device)

        optimizer.zero_grad()
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            policy_label=policy_label,
            violation_label=violation_label,
            stage=stage,
        )

        loss = out["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_logits.append(out["violation_logit"].detach().cpu())
        all_labels.append(violation_label.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader)
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, stage="backbone"):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc=f"Eval [{stage}]", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        visual_features = batch["visual_features"].to(device)
        policy_label = batch["policy_label"].to(device)
        violation_label = batch["violation_label"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            policy_label=policy_label,
            violation_label=violation_label,
            stage=stage,
        )

        total_loss += out["loss"].item()
        all_logits.append(out["violation_logit"].detach().cpu())
        all_labels.append(violation_label.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def build_rag_index(model, loader, device):
    """Collect violation embeddings from training set for RAG index."""
    model.eval()
    violation_embeds = []
    violation_labels_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Building RAG index"):
            viol_mask = batch["violation_label"] == 1
            if viol_mask.sum() == 0:
                continue

            input_ids = batch["input_ids"][viol_mask].to(device)
            attention_mask = batch["attention_mask"][viol_mask].to(device)
            visual_features = batch["visual_features"][viol_mask].to(device)
            policy_label = batch["policy_label"][viol_mask].to(device)

            out = model.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=visual_features,
                policy_label=policy_label,
            )
            violation_embeds.append(out["pooled"].cpu())
            violation_labels_list.append(batch["violation_label"][viol_mask])

    if violation_embeds:
        all_embeds = torch.cat(violation_embeds, dim=0)
        all_labels = torch.cat(violation_labels_list, dim=0)
        model.rag.build_index(all_embeds.to(device), all_labels.to(device))
        print(f"RAG index built with {all_embeds.shape[0]} violation examples")
    else:
        print("No violation examples found for RAG index")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"LM backbone: {args.lm_model}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Generating synthetic dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = generate_synthetic_dataset(args.n_samples, args.violation_ratio, args.seed)
    n_train = int(0.8 * len(samples))
    dataset = ModerationDataset(samples, tokenizer, args.max_caption_len, args.visual_dim)
    train_ds, val_ds = random_split(dataset, [n_train, len(samples) - n_train])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Initializing UNIVID system...")
    model = UNIVIDSystem(
        lm_model_name=args.lm_model,
        visual_dim=args.visual_dim,
        n_policies=len(POLICY_CATEGORIES),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    history = {"backbone": [], "lite": []}

    # ── Phase 1: Train backbone ───────────────────────────────────────────────
    print("\n=== Phase 1: Training UNIVID Backbone ===")
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_backbone
    )

    best_leakage = float("inf")
    for epoch in range(args.epochs_backbone):
        train_m = train_epoch(model, train_loader, optimizer, device, stage="backbone")
        val_m = evaluate(model, val_loader, device, stage="backbone")
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{args.epochs_backbone} | "
            f"train_loss={train_m['loss']:.4f} | "
            f"val_f1={val_m['f1']:.4f} | "
            f"val_leakage={val_m['violation_leakage']:.4f} | "
            f"val_overkill={val_m['overkill_rate']:.4f}"
        )
        history["backbone"].append({"epoch": epoch + 1, "train": train_m, "val": val_m})

        if val_m["violation_leakage"] < best_leakage:
            best_leakage = val_m["violation_leakage"]
            torch.save(model.backbone.state_dict(),
                       f"{args.output_dir}/backbone_best.pt")

    # ── Phase 2: Build RAG index ───────────────────────────────────────────────
    print("\n=== Phase 2: Building RAG Index from Training Violations ===")
    model.backbone.load_state_dict(
        torch.load(f"{args.output_dir}/backbone_best.pt", map_location=device)
    )
    build_rag_index(model, train_loader, device)

    # ── Phase 3: Train UNIVID-Lite ────────────────────────────────────────────
    print("\n=== Phase 3: Training UNIVID-Lite ===")
    # Freeze backbone during Lite training
    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer_lite = torch.optim.AdamW(model.lite.parameters(), lr=args.lr * 2)

    for epoch in range(args.epochs_lite):
        train_m = train_epoch(model, train_loader, optimizer_lite, device, stage="lite")
        val_m = evaluate(model, val_loader, device, stage="lite")

        print(
            f"Lite Epoch {epoch+1}/{args.epochs_lite} | "
            f"train_loss={train_m['loss']:.4f} | "
            f"val_f1={val_m['f1']:.4f} | "
            f"val_leakage={val_m['violation_leakage']:.4f}"
        )
        history["lite"].append({"epoch": epoch + 1, "train": train_m, "val": val_m})

    # ── Phase 4: Initialize Trend Head prototypes ─────────────────────────────
    print("\n=== Phase 4: Initializing Trend Governance Prototypes ===")
    model.eval()
    with torch.no_grad():
        all_embeds = []
        all_viol = []
        for batch in train_loader:
            out = model.backbone(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                visual_features=batch["visual_features"].to(device),
            )
            all_embeds.append(out["pooled"].cpu())
            all_viol.append(batch["violation_label"])
        all_embeds = torch.cat(all_embeds, dim=0)
        all_viol = torch.cat(all_viol, dim=0)

        viol_embeds = all_embeds[all_viol == 1].to(device)
        if len(viol_embeds) > 0:
            model.trend.update_prototypes(viol_embeds, momentum=0.0)
            print(f"Trend prototypes initialized from {len(viol_embeds)} violation examples")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    torch.save(model.state_dict(), f"{args.output_dir}/univid_system.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        # Convert tensors to Python scalars for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, torch.Tensor):
                return obj.item()
            return obj
        json.dump(make_serializable(history), f, indent=2)

    print(f"\n✓ Training complete. Checkpoint saved to {args.output_dir}/")
    print(f"  Best backbone violation_leakage: {best_leakage:.4f}")


if __name__ == "__main__":
    main()
