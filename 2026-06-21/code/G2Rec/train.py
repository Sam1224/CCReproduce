"""
G2Rec training script (arXiv 2606.20554).

Usage:
    python train.py [--n_items 1000] [--epochs 10] [--batch_size 64]

This reproduces the training pipeline from the paper:
  - Holistic co-engagement graph → interest prototype logits
  - Supervised tokenization loss (Eq. §3.3)
  - Autoregressive recommendation loss (Eq. §3.4)
  - Total: L = L_rec + λ·L_tok (Eq. §3.5)
"""

import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from data import generate_toy_dataset, get_dataloaders
from model import G2Rec


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    print("Generating toy dataset...")
    data = generate_toy_dataset(
        n_users=args.n_users,
        n_items=args.n_items,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    train_loader, val_loader = get_dataloaders(
        data, batch_size=args.batch_size, val_split=0.1
    )

    # ---- Model ----
    model = G2Rec(
        n_items=args.n_items,
        embed_dim=args.embed_dim,
        codebook_size=args.codebook_size,
        n_codes=args.n_codes,
        n_prototypes=args.n_prototypes,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lam=args.lam,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"G2Rec parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    # ---- Training loop ----
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss_sum = rec_loss_sum = tok_loss_sum = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for user_seqs, targets in pbar:
            user_seqs = user_seqs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            total_loss, rec_loss, tok_loss = model(user_seqs, targets)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss_sum += total_loss.item()
            rec_loss_sum += rec_loss.item()
            tok_loss_sum += tok_loss.item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                rec=f"{rec_loss.item():.4f}",
                tok=f"{tok_loss.item():.4f}",
            )

        scheduler.step()

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for user_seqs, targets in val_loader:
                user_seqs = user_seqs.to(device)
                targets = targets.to(device)
                loss, _, _ = model(user_seqs, targets)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        avg_train = total_loss_sum / max(n_batches, 1)
        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={avg_train:.4f} "
            f"(rec={rec_loss_sum/n_batches:.4f}, "
            f"tok={tok_loss_sum/n_batches:.4f}) | "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "g2rec_best.pt")
            print(f"  → Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="G2Rec training")
    # Data
    p.add_argument("--n_users", type=int, default=500)
    p.add_argument("--n_items", type=int, default=1000)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    # Model
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--n_codes", type=int, default=4)
    p.add_argument("--n_prototypes", type=int, default=32)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--lam", type=float, default=0.5, help="λ: tok_loss weight")
    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
