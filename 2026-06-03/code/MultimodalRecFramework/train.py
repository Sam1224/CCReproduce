"""
Training script for the Tripartite Multimodal Recommendation Framework.

Paper: arXiv:2605.09338 (SIGIR 2026) — Meta Platforms

Usage:
    python train.py --epochs 10 --batch_size 256 --lr 1e-4

For production use at Meta scale:
  - visual_dim=2048 (ResNet-50 or ViT-L)
  - text_dim=4096 (LLaMA2 7B hidden size)
  - embed_dim=256
  - Offline pre-compute captions for 1B+ items
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import TripartiteRecFramework


def generate_toy_data(n: int, visual_dim: int, text_dim: int, vocab_size: int, caption_len: int):
    """Generate synthetic training data for testing."""
    user_feats = torch.randn(n, text_dim)
    pos_visual = torch.randn(n, visual_dim)
    pos_text = torch.randn(n, text_dim)
    neg_visual = torch.randn(n, visual_dim)
    neg_text = torch.randn(n, text_dim)
    captions = torch.randint(1, vocab_size, (n, caption_len))
    return TensorDataset(user_feats, pos_visual, pos_text, neg_visual, neg_text, captions)


def train(args):
    device = torch.device(args.device)
    print(f"Tripartite Multimodal Recommendation Framework Training")
    print(f"  visual_dim={args.visual_dim}, text_dim={args.text_dim}, embed_dim={args.embed_dim}")
    print(f"  device={device}")
    print()

    model = TripartiteRecFramework(
        visual_dim=args.visual_dim,
        text_dim=args.text_dim,
        embed_dim=args.embed_dim,
        caption_vocab_size=args.vocab_size,
        max_caption_len=args.caption_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    dataset = generate_toy_data(
        args.train_size, args.visual_dim, args.text_dim, args.vocab_size, args.caption_len
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {"bpr": 0, "caption": 0, "alignment": 0, "total": 0}
        n_batches = 0

        for batch in loader:
            user_f, pos_v, pos_t, neg_v, neg_t, captions = [b.to(device) for b in batch]

            losses = model(
                user_features=user_f,
                pos_visual_features=pos_v,
                pos_text_features=pos_t,
                neg_visual_features=neg_v,
                neg_text_features=neg_t,
                pos_caption_ids=captions,
            )

            optimizer.zero_grad()
            losses["total_loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in epoch_losses:
                key = k + "_loss"
                if key in losses:
                    epoch_losses[k] += losses[key].item()
            n_batches += 1

        scheduler.step()
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"total={epoch_losses['total']:.4f} | "
            f"bpr={epoch_losses['bpr']:.4f} | "
            f"caption={epoch_losses['caption']:.4f} | "
            f"align={epoch_losses['alignment']:.4f}"
        )

    print()
    print("Training complete.")
    print("In production (Meta): offline pre-compute item embeddings → store in Faiss index")
    print("Online serving: user_encoder(query) → ANN retrieval → score → rank")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--visual_dim", type=int, default=128)  # small for toy
    p.add_argument("--text_dim", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--vocab_size", type=int, default=1000)
    p.add_argument("--caption_len", type=int, default=16)
    p.add_argument("--train_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
