"""
UNIVID training script.

Loss = λ_cap * L_caption + λ_viol * L_violation

where:
  L_caption  = weighted cross-entropy over caption tokens  [UNIVID §3.3]
  L_violation = weighted BCE on violation label            [UNIVID §3.4]

Human-labeled samples get weight w=2.0; synthetic labels get w=1.0.
This follows the UNIVID mixed-label training recipe.
"""

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import UNIVIDDataset, generate_dataset, W2I
from model import UNIVID


# ── loss helpers ──────────────────────────────────────────────────────────────

def caption_loss(
    caption_logits: torch.Tensor,  # (B, T, V)
    caption_ids: torch.Tensor,     # (B, T)
    label_weight: torch.Tensor,    # (B,)
    pad_id: int = W2I["<PAD>"],
) -> torch.Tensor:
    """
    Weighted next-token prediction loss.
    Shift by 1: predict tokens 1..T from tokens 0..T-1.  [UNIVID §3.2]
    """
    B, T, V = caption_logits.shape
    logits = caption_logits[:, :-1, :].reshape(-1, V)    # (B*(T-1), V)
    targets = caption_ids[:, 1:].reshape(-1)             # (B*(T-1),)

    per_token_loss = F.cross_entropy(logits, targets, ignore_index=pad_id, reduction="none")
    # Expand label_weight to per-token
    per_token_weight = label_weight.unsqueeze(1).expand(B, T - 1).reshape(-1)
    return (per_token_loss * per_token_weight).mean()


def violation_loss(
    viol_logit: torch.Tensor,     # (B,)
    violation: torch.Tensor,      # (B,)  float 0/1
    label_weight: torch.Tensor,   # (B,)
) -> torch.Tensor:
    """Weighted binary cross-entropy on violation label. [UNIVID §3.4]"""
    per_sample = F.binary_cross_entropy_with_logits(
        viol_logit, violation, reduction="none"
    )
    return (per_sample * label_weight).mean()


# ── training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    if os.path.exists(args.data):
        raw = torch.load(args.data)
        dataset = UNIVIDDataset(raw)
    else:
        print("No dataset file found; generating synthetic data …")
        dataset = generate_dataset(n_videos=500)

    n_train = int(0.85 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # ── Model ──
    model = UNIVID(
        frame_dim=64,
        hidden_dim=128,
        n_heads=4,
        max_seq=12,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            video_feat   = batch["video_feat"].to(device)
            policy_id    = batch["policy_id"].to(device)
            caption_ids  = batch["caption_ids"].to(device)
            violation    = batch["violation"].to(device)
            label_weight = batch["label_weight"].to(device)

            out = model(video_feat, policy_id, caption_ids)

            l_cap  = caption_loss(out["caption_logits"], caption_ids, label_weight)
            l_viol = violation_loss(out["viol_logit"], violation, label_weight)
            loss = args.lambda_cap * l_cap + args.lambda_viol * l_viol

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        scheduler.step()
        avg_train = total_loss / len(train_loader)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for batch in val_loader:
                video_feat   = batch["video_feat"].to(device)
                policy_id    = batch["policy_id"].to(device)
                caption_ids  = batch["caption_ids"].to(device)
                violation    = batch["violation"].to(device)
                label_weight = batch["label_weight"].to(device)

                out = model(video_feat, policy_id, caption_ids)
                l_cap  = caption_loss(out["caption_logits"], caption_ids, label_weight)
                l_viol = violation_loss(out["viol_logit"], violation, label_weight)
                val_loss += (args.lambda_cap * l_cap + args.lambda_viol * l_viol).item()

                preds = (out["viol_logit"] > 0.0).long()
                n_correct += (preds == violation.long()).sum().item()
                n_total += preds.size(0)

        avg_val = val_loss / len(val_loader)
        acc = n_correct / n_total
        print(
            f"Epoch {epoch:3d} | train_loss={avg_train:.4f} "
            f"val_loss={avg_val:.4f} val_acc={acc:.4f}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
            print(f"  ✓ Saved best model → {args.out}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        type=str,   default="data/univid_toy.pt")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--lambda-cap",  type=float, default=1.0,
                        help="Weight on caption generation loss [UNIVID §3.3]")
    parser.add_argument("--lambda-viol", type=float, default=2.0,
                        help="Weight on violation classification loss [UNIVID §3.4]")
    parser.add_argument("--out",         type=str,   default="runs/univid.pt")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    train(args)
