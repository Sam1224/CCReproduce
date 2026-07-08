"""
YuvionVL — Training Script

Implements:
1. Standard SFT (supervised fine-tuning) on safety data
2. C2FT (Confuse-then-Contrast Fine-Tuning) on confusion pairs

Usage:
    python train.py --mode sft --epochs 5 --batch_size 8
    python train.py --mode c2ft --epochs 3 --batch_size 8

Paper: "Yuvion VL: A Multimodal Foundation Model for Adversarial Content and AI Safety"
arXiv: https://arxiv.org/abs/2606.25034
"""

import argparse
import os
import random
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data import SafetyDataset, ConfusionPairDataset
from model import YuvionVLModel, C2FTLoss, SafetyGate


# ─── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_safety(batch: list[dict], vocab_size: int = 32000,
                   max_len: int = 64) -> dict:
    """Toy collator: tokenize text via simple character-level encoding."""
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    # Character-level tokenization (toy)
    def tokenize(text: str, max_len: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [min(ord(c), vocab_size - 1) for c in text[:max_len]]
        pad_len = max_len - len(ids)
        mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [0] * pad_len
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    input_ids, attention_masks = zip(*[tokenize(t, max_len) for t in texts])
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "pixel_values": None,    # omit image loading for toy reproduction
        "labels": labels,
    }


def collate_c2ft(batch: list[dict], max_len: int = 64) -> dict:
    """Collator for confusion pairs (C2FT training)."""
    anchor_texts = [item["anchor_text"] for item in batch]
    confused_texts = [item["confused_text"] for item in batch]
    anchor_labels = torch.tensor([item["anchor_label"] for item in batch], dtype=torch.long)

    def tokenize(text: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [min(ord(c), 31999) for c in text[:max_len]]
        pad = max_len - len(ids)
        return (
            torch.tensor(ids + [0] * pad, dtype=torch.long),
            torch.tensor([1] * len(ids) + [0] * pad, dtype=torch.long),
        )

    anc_ids, anc_masks = zip(*[tokenize(t) for t in anchor_texts])
    con_ids, con_masks = zip(*[tokenize(t) for t in confused_texts])

    return {
        "anchor_input_ids": torch.stack(anc_ids),
        "anchor_attention_mask": torch.stack(anc_masks),
        "confused_input_ids": torch.stack(con_ids),
        "confused_attention_mask": torch.stack(con_masks),
        "anchor_labels": anchor_labels,
    }


# ─── SFT Training ─────────────────────────────────────────────────────────────

def train_sft(
    model: YuvionVLModel,
    dataset: SafetyDataset,
    num_epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cpu",
) -> dict:
    """Standard supervised fine-tuning on safety classification data."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_safety,
    )

    history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in loader:
            optimizer.zero_grad()

            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"],
                labels=batch["labels"].to(device),
            )

            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        history.append(avg_loss)
        print(f"  [SFT] Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.4f}")

    return {"sft_history": history}


# ─── C2FT Training ────────────────────────────────────────────────────────────

def train_c2ft(
    model: YuvionVLModel,
    base_dataset: SafetyDataset,
    num_epochs: int = 3,
    batch_size: int = 4,
    lr: float = 5e-5,
    temperature: float = 0.07,
    alpha: float = 0.5,
    device: str = "cpu",
) -> dict:
    """
    Confuse-then-Contrast Fine-Tuning.

    Stage description (from paper):
    1. Mine confusion pairs: identify model-specific confusions
       (pairs where the model assigns high confidence to the wrong class)
    2. Apply joint contrastive supervision across multiple confused pairs
    3. Combined with cross-entropy on anchor examples

    This pushes apart representations of (anchor, confused) pairs,
    improving fine-grained discrimination on adversarial examples.
    """
    model = model.to(device)
    confusion_dataset = ConfusionPairDataset(base_dataset)

    if len(confusion_dataset) == 0:
        print("  [C2FT] No confusion pairs available. Skipping.")
        return {}

    loader = DataLoader(
        confusion_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_c2ft,
    )

    c2ft_loss_fn = C2FTLoss(temperature=temperature, alpha=alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_ct = 0.0
        num_batches = 0

        for batch in loader:
            optimizer.zero_grad()

            anc_ids = batch["anchor_input_ids"].to(device)
            anc_mask = batch["anchor_attention_mask"].to(device)
            con_ids = batch["confused_input_ids"].to(device)
            con_mask = batch["confused_attention_mask"].to(device)
            anc_labels = batch["anchor_labels"].to(device)

            # Encode both anchor and confused examples
            anc_feats = model.encode(anc_ids, anc_mask)
            con_feats = model.encode(con_ids, con_mask)

            # Safety logits for anchor (for CE loss)
            anc_logits = model.safety_head(anc_feats)

            # C2FT loss: CE + contrastive push-apart
            loss_dict = c2ft_loss_fn(anc_feats, con_feats, anc_labels, anc_logits)
            loss = loss_dict["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ce += loss_dict["ce_loss"]
            epoch_ct += loss_dict["contrastive_loss"]
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_ce = epoch_ce / max(num_batches, 1)
        avg_ct = epoch_ct / max(num_batches, 1)
        history.append(avg_loss)
        print(f"  [C2FT] Epoch {epoch+1}/{num_epochs} — "
              f"total: {avg_loss:.4f}, ce: {avg_ce:.4f}, contrastive: {avg_ct:.4f}")

    return {"c2ft_history": history}


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: YuvionVLModel,
    dataset: SafetyDataset,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_safety)

    total = 0
    correct = 0
    tp = fp = tn = fn = 0

    for batch in loader:
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        preds = output["safety_logits"].argmax(dim=-1).cpu()
        labels = batch["labels"]

        correct += (preds == labels).sum().item()
        total += len(labels)

        for p, l in zip(preds.tolist(), labels.tolist()):
            if l == 1 and p == 1: tp += 1
            elif l == 0 and p == 1: fp += 1
            elif l == 0 and p == 0: tn += 1
            else: fn += 1

    accuracy = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": total,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YuvionVL Training")
    parser.add_argument("--mode", choices=["sft", "c2ft", "both"], default="both")
    parser.add_argument("--epochs_sft", type=int, default=5)
    parser.add_argument("--epochs_c2ft", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=256)  # small for toy
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = SafetyDataset(augment=True)
    print(f"Dataset size: {len(dataset)} examples")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = YuvionVLModel(
        embed_dim=args.embed_dim,
        num_violation_types=5,
        visual_depth=2,   # toy
        text_layers=2,    # toy
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # ── SFT Phase ─────────────────────────────────────────────────────────────
    if args.mode in ("sft", "both"):
        print("\n=== Phase 1: Supervised Fine-Tuning (SFT) ===")
        sft_result = train_sft(
            model, dataset,
            num_epochs=args.epochs_sft,
            batch_size=args.batch_size,
            lr=args.lr,
            device=str(device),
        )
        metrics_after_sft = evaluate(model, dataset, device=str(device))
        print(f"  After SFT — Accuracy: {metrics_after_sft['accuracy']:.4f}, "
              f"F1: {metrics_after_sft['f1']:.4f}")

    # ── C2FT Phase ────────────────────────────────────────────────────────────
    if args.mode in ("c2ft", "both"):
        print("\n=== Phase 2: Confuse-then-Contrast Fine-Tuning (C2FT) ===")
        c2ft_result = train_c2ft(
            model, dataset,
            num_epochs=args.epochs_c2ft,
            batch_size=min(args.batch_size, 4),
            lr=args.lr * 0.5,
            device=str(device),
        )
        metrics_after_c2ft = evaluate(model, dataset, device=str(device))
        print(f"  After C2FT — Accuracy: {metrics_after_c2ft['accuracy']:.4f}, "
              f"F1: {metrics_after_c2ft['f1']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/yuvion_vl_toy.pt")
    print("\nModel saved to checkpoints/yuvion_vl_toy.pt")

    # ── Safety Gate Demo ──────────────────────────────────────────────────────
    print("\n=== Safety Gate Demo ===")
    gate = SafetyGate(model, threshold=0.4)

    def quick_tokenize(text: str, max_len: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [min(ord(c), 31999) for c in text[:max_len]]
        pad = max_len - len(ids)
        ids_t = torch.tensor(ids + [0] * pad, dtype=torch.long).unsqueeze(0)
        mask_t = torch.tensor([1]*len(ids) + [0]*pad, dtype=torch.long).unsqueeze(0)
        return ids_t, mask_t

    test_texts = [
        "正品 Nike 运动鞋，官方旗舰店直发",
        "正品 Nikke 运动鞋，限时特卖！",  # suspicious name
        "原价9999元，现价仅需99元！限时秒杀！",  # price fraud
    ]
    for text in test_texts:
        ids, mask = quick_tokenize(text)
        result = gate.check(ids, mask)
        print(f"  '{text[:40]}...' → {result[0]}")


if __name__ == "__main__":
    main()
