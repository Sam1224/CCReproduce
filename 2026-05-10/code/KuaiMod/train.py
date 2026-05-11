"""
KuaiMod Training Script

Two-stage training as described in the paper (Section 3.2):

Stage 1 — Classification Warmup:
  - Train classifier head with visual + text features
  - No CoT generation
  - Faster convergence for basic policy alignment

Stage 2 — CoT Fine-tuning:
  - End-to-end training: CoT decoder + classifier
  - Loss = λ_cot * L_cot + λ_clf * L_clf   (λ_cot=0.5, λ_clf=1.0)
  - Paper uses curriculum: easy cases first, then borderline

Usage:
  python train.py --stage warmup --epochs 3
  python train.py --stage cot --epochs 5 --checkpoint checkpoints/warmup_best.pt
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import SVPDataset, SVPCollator, ID2LABEL
from model import KuaiMod


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simple_tokenize(texts, vocab_size=1000, max_len=64):
    """Toy tokenizer: maps each char to an int, pads to max_len."""
    ids = []
    for text in texts:
        toks = [ord(c) % (vocab_size - 1) + 1 for c in text[:max_len]]
        toks = toks + [0] * (max_len - len(toks))
        ids.append(toks)
    return torch.tensor(ids, dtype=torch.long)


def simple_tokenize_cot(cots, vocab_size=1000, max_len=32):
    return simple_tokenize(cots, vocab_size, max_len)


def compute_metrics(all_preds, all_labels):
    from collections import defaultdict
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = correct / len(all_labels)

    # Per-class recall
    class_tp = defaultdict(int)
    class_total = defaultdict(int)
    for p, l in zip(all_preds, all_labels):
        class_total[l] += 1
        if p == l:
            class_tp[l] += 1

    recalls = {ID2LABEL[k]: class_tp[k] / max(class_total[k], 1)
               for k in class_total}
    return {"accuracy": acc, "per_class_recall": recalls}


def train_epoch(model, loader, optimizer, device, stage, vocab_size):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc=f"Train [{stage}]"):
        frames = batch["frames"].to(device)
        texts = batch["texts"]
        labels = batch["labels"].to(device)
        cot_texts = batch["cot_rationales"]

        text_ids = simple_tokenize(texts, vocab_size).to(device)
        cot_ids = simple_tokenize_cot(cot_texts, vocab_size).to(device) \
            if stage == "cot" else None

        optimizer.zero_grad()
        out = model(frames, text_ids, cot_ids=cot_ids, labels=labels, stage=stage)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = out["clf_logits"].argmax(-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_preds, all_labels)
    return total_loss / len(loader), metrics


@torch.no_grad()
def eval_epoch(model, loader, device, stage, vocab_size):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Eval"):
        frames = batch["frames"].to(device)
        texts = batch["texts"]
        labels = batch["labels"].to(device)
        cot_texts = batch["cot_rationales"]

        text_ids = simple_tokenize(texts, vocab_size).to(device)
        cot_ids = simple_tokenize_cot(cot_texts, vocab_size).to(device) \
            if stage == "cot" else None

        out = model(frames, text_ids, cot_ids=cot_ids, labels=labels, stage=stage)
        if "loss" in out:
            total_loss += out["loss"].item()

        preds = out["clf_logits"].argmax(-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_preds, all_labels)
    return total_loss / max(len(loader), 1), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["warmup", "cot"], default="warmup")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to load weights (for stage 2)")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    dataset = SVPDataset(num_samples=args.num_samples, seed=args.seed)
    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))

    collator = SVPCollator()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collator)

    # Model
    model = KuaiMod(vocab_size=args.vocab_size).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.stage == "cot":
        # Freeze visual encoder during CoT stage (paper uses curriculum warmup)
        for param in model.visual_encoder.parameters():
            param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, device, args.stage, args.vocab_size
        )
        val_loss, val_metrics = eval_epoch(
            model, val_loader, device, args.stage, args.vocab_size
        )
        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss: {train_loss:.4f}  acc: {train_metrics['accuracy']:.3f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_metrics['accuracy']:.3f}"
        )
        print(f"  Val per-class recall: {val_metrics['per_class_recall']}")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            ckpt_path = save_dir / f"best_{args.stage}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best checkpoint → {ckpt_path}")

    print(f"\nTraining complete. Best val accuracy: {best_acc:.3f}")


if __name__ == "__main__":
    main()
