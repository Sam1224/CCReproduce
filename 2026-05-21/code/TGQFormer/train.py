"""Training script for TGQ-Former (toy reproduction of arXiv:2605.17366).

Usage:
    python train.py --epochs 10 --batch_size 64 --output_dir runs/tgqformer
"""
import argparse
import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from tgqformer import TGQFormer, ECommerceDataset, ContrastiveLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--num_products", type=int, default=200)
    p.add_argument("--images_per_product", type=int, default=4)
    p.add_argument("--noise_ratio", type=float, default=0.35)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--output_dir", type=str, default="runs/tgqformer")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Dataset
    full_ds = ECommerceDataset(
        num_products=args.num_products,
        images_per_product=args.images_per_product,
        noise_ratio=args.noise_ratio,
    )
    n_train = int(0.8 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [n_train, len(full_ds) - n_train])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = TGQFormer(
        d_model=args.d_model,
        emb_dim=args.emb_dim,
        num_products=args.num_products if hasattr(args, "num_products") else 200,
    ).to(device)

    criterion = ContrastiveLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for batch in train_dl:
            optimizer.zero_grad()
            out = model(
                batch["q_image"].to(device),  batch["q_text"].to(device),
                batch["pos_image"].to(device), batch["pos_text"].to(device),
                batch["neg_image"].to(device), batch["neg_text"].to(device),
            )
            loss_dict = criterion(out["q"], out["pos"], out["neg"])
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss_dict["loss"].item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_dl)

        # --- Val loss ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                out = model(
                    batch["q_image"].to(device),  batch["q_text"].to(device),
                    batch["pos_image"].to(device), batch["pos_text"].to(device),
                    batch["neg_image"].to(device), batch["neg_text"].to(device),
                )
                val_loss += criterion(out["q"], out["pos"], out["neg"])["loss"].item()
        val_loss /= max(len(val_dl), 1)

        history.append({"epoch": epoch, "train_loss": avg_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:3d} | train_loss={avg_loss:.4f} | val_loss={val_loss:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
