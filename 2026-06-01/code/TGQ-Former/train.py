"""Training script for TGQ-Former (contrastive I2I learning)."""
import argparse
import os
import torch
import torch.optim as optim

from model import TGQFormer, infonce_loss
from dataset import get_dataloaders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_queries", type=int, default=16)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    return p.parse_args()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        va = batch["visual_a"].to(device)
        mia = batch["meta_ids_a"].to(device)
        mma = batch["meta_mask_a"].to(device)
        vb = batch["visual_b"].to(device)
        mib = batch["meta_ids_b"].to(device)
        mmb = batch["meta_mask_b"].to(device)

        optimizer.zero_grad()
        logits, emb_a, emb_b = model(va, mia, vb, mib, mma, mmb)
        loss = infonce_loss(logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        va = batch["visual_a"].to(device)
        mia = batch["meta_ids_a"].to(device)
        mma = batch["meta_mask_a"].to(device)
        vb = batch["visual_b"].to(device)
        mib = batch["meta_ids_b"].to(device)
        mmb = batch["meta_mask_b"].to(device)
        logits, _, _ = model(va, mia, vb, mib, mma, mmb)
        total_loss += infonce_loss(logits).item()
    return total_loss / len(loader)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, _, _ = get_dataloaders(args.batch_size)

    model = TGQFormer(
        visual_dim=2048,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        embed_dim=args.embed_dim,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:3d} | Train {train_loss:.4f} | Val {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.save_dir}/best.pt")

    print(f"\nBest val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
