"""
Training script for TGQ-Former (arXiv: 2605.17366)
E-Commerce Item-to-Item Retrieval with Text-Guided Dual-Stream Q-Former

Usage:
    python train.py --epochs 20 --batch_size 128 --alpha_r3 0.1
"""

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import TGQFormerRetrieval, TGQFormerLoss


# ---------------------------------------------------------------------------
# Toy Dataset
# ---------------------------------------------------------------------------

class EComI2IDataset(Dataset):
    """
    E-commerce I2I retrieval dataset.
    Each example: (query_item, positive_item) pair for contrastive learning.
    Negative samples: in-batch negatives.

    Features:
        text_feat: Product metadata embedding (title + attributes concatenated, BERT-encoded)
        visual_feat: Product image patch features (ViT/CLIP visual tokens)
    """

    def __init__(
        self,
        num_samples: int = 2000,
        text_dim: int = 768,
        visual_dim: int = 1024,
        num_visual_tokens: int = 49,  # 7x7 patch grid (simulated)
    ):
        self.num_samples = num_samples
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.num_visual_tokens = num_visual_tokens

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Query item
        query_text = torch.randn(self.text_dim)
        query_visual = torch.randn(self.num_visual_tokens, self.visual_dim)

        # Positive item: correlated with query (same category, similar style)
        # In practice: real product pairs; here: add small noise to query features
        pos_text = query_text + torch.randn_like(query_text) * 0.1
        pos_visual = query_visual + torch.randn_like(query_visual) * 0.1

        return {
            "query_text": query_text,
            "query_visual": query_visual,
            "pos_text": pos_text,
            "pos_visual": pos_visual,
        }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("TGQ-Former — E-Commerce I2I Retrieval Training")
    print("arXiv: 2605.17366")
    print("=" * 60)

    model = TGQFormerRetrieval(
        text_dim=args.text_dim,
        visual_dim=args.visual_dim,
        num_visual_tokens=args.num_visual_tokens,
        query_dim=args.query_dim,
        num_meta_queries=args.num_meta_queries,
        num_explore_queries=args.num_explore_queries,
        num_layers=args.num_layers,
        embedding_dim=args.embedding_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"  Dual-stream Q-Former layers: {args.num_layers}")
    print(f"  Meta queries: {args.num_meta_queries}, Explore queries: {args.num_explore_queries}")
    print(f"  DGVM + R³ regularizer: enabled")

    dataset = EComI2IDataset(
        num_samples=args.num_samples,
        text_dim=args.text_dim,
        visual_dim=args.visual_dim,
        num_visual_tokens=args.num_visual_tokens,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = TGQFormerLoss(alpha_r3=args.alpha_r3)

    print(f"\nTraining for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_contrastive = 0.0
        total_r3 = 0.0

        for batch in loader:
            query_text = batch["query_text"].to(device)
            query_visual = batch["query_visual"].to(device)
            pos_text = batch["pos_text"].to(device)
            pos_visual = batch["pos_visual"].to(device)

            contrastive_loss, r3_penalty = model(
                query_text, query_visual, pos_text, pos_visual
            )

            losses = criterion(contrastive_loss, r3_penalty)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += losses["total"].item()
            total_contrastive += losses["contrastive"].item()
            total_r3 += losses["r3"].item()

        scheduler.step()
        n = len(loader)
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Total: {total_loss/n:.4f} | "
            f"Contrastive: {total_contrastive/n:.4f} | "
            f"R³: {total_r3/n:.4f}"
        )

    save_path = os.path.join(args.output_dir, "tgq_former.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    # Quick evaluation: retrieval at k on a small held-out set
    print("\n[Evaluation] Running Recall@K on held-out set...")
    evaluate(model, device, args)


def evaluate(model, device, args):
    """Evaluate I2I retrieval recall@K."""
    model.eval()
    eval_dataset = EComI2IDataset(num_samples=200, text_dim=args.text_dim,
                                   visual_dim=args.visual_dim, num_visual_tokens=args.num_visual_tokens)
    eval_loader = DataLoader(eval_dataset, batch_size=200)

    with torch.no_grad():
        batch = next(iter(eval_loader))
        q_text = batch["query_text"].to(device)
        q_vis = batch["query_visual"].to(device)
        p_text = batch["pos_text"].to(device)
        p_vis = batch["pos_visual"].to(device)

        q_emb, _ = model.encode_item(q_text, q_vis)
        p_emb, _ = model.encode_item(p_text, p_vis)

        # Similarity matrix
        sim = torch.matmul(q_emb, p_emb.T)  # (N, N)
        ranked = sim.argsort(dim=-1, descending=True)
        targets = torch.arange(len(q_emb), device=device)

        for k in [1, 5, 10]:
            recall = (ranked[:, :k] == targets.unsqueeze(1)).any(dim=1).float().mean()
            print(f"  Recall@{k}: {recall.item():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train TGQ-Former for E-Commerce I2I Retrieval")
    parser.add_argument("--num_samples", type=int, default=4000)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--visual_dim", type=int, default=1024)
    parser.add_argument("--num_visual_tokens", type=int, default=49)
    parser.add_argument("--query_dim", type=int, default=256)
    parser.add_argument("--num_meta_queries", type=int, default=16)
    parser.add_argument("--num_explore_queries", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha_r3", type=float, default=0.1, help="R³ loss weight")
    parser.add_argument("--output_dir", default="checkpoints/")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
