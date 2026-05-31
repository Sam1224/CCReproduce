"""
Training script for SARM: LLM-Augmented Semantic Anchor for Live-Streaming Ranking
arXiv: 2602.09401  |  Kuaishou Technology

Usage:
    python train.py --num_anchors 1000 --anchor_tokens 8 --epochs 10
"""

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import SARMRankingModel, LLMSemanticAnchorGenerator, SARMTrainingLoss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LiveStreamRankingDataset(Dataset):
    """
    Live-stream ranking dataset.
    Each example: (stream_id, user_id) → engagement label (watch/click)

    Expected format (simplified toy version):
        stream features: text, visual, interaction
        user features: user embedding
        context features: time, device
        label: 1 (watched) or 0 (skipped)
    """

    def __init__(self, num_samples: int = 1000, split: str = "train"):
        self.num_samples = num_samples
        # Toy data dimensions matching model architecture
        self.text_dim = 768
        self.visual_dim = 2048
        self.interaction_dim = 64
        self.user_dim = 128
        self.context_dim = 64

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "text_feat": torch.randn(self.text_dim),
            "visual_feat": torch.randn(self.visual_dim),
            "interaction_feat": torch.randn(self.interaction_dim),
            "user_feat": torch.randn(self.user_dim),
            "context_feat": torch.randn(self.context_dim),
            # For BPR training: positive and negative samples paired
            "neg_text_feat": torch.randn(self.text_dim),
            "neg_visual_feat": torch.randn(self.visual_dim),
            "neg_interaction_feat": torch.randn(self.interaction_dim),
            "label": torch.tensor(float(idx % 2 == 0)),
        }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SARM — Semantic Anchor Live-Streaming Ranking")
    print("arXiv: 2602.09401 | Kuaishou Technology")
    print("=" * 60)

    # Stage 1: Generate semantic anchor descriptions (offline, LLM-based)
    print("\n[Stage 1] Generating semantic anchor initializations (offline LLM)...")
    anchor_generator = LLMSemanticAnchorGenerator()
    # In practice: iterate over all live-stream sessions and generate descriptions
    # Here: toy demonstration
    toy_transcript = "今天给大家推荐这款网红爆款连衣裙，限时特价！"
    anchor_desc = anchor_generator.generate_anchor_description(toy_transcript)
    print(f"  Sample anchor description: {anchor_desc[:80]}...")

    # Stage 2: End-to-end ranking training with learnable anchors
    print(f"\n[Stage 2] End-to-end training with semantic anchors...")

    model = SARMRankingModel(
        num_anchor_tokens=args.anchor_tokens,
        content_dim=256,
        user_dim=128,
        context_dim=64,
        anchor_hidden_dim=128,
    ).to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Anchor parameters: {sum(p.numel() for p in model.semantic_anchor.parameters()):,}")

    train_dataset = LiveStreamRankingDataset(num_samples=args.num_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Separate LR for semantic anchors vs. rest of model (anchors need higher LR)
    anchor_params = list(model.semantic_anchor.parameters())
    other_params = [p for n, p in model.named_parameters() if "semantic_anchor" not in n]

    optimizer = optim.AdamW([
        {"params": anchor_params, "lr": args.anchor_lr},
        {"params": other_params, "lr": args.model_lr},
    ], weight_decay=1e-4)

    criterion = SARMTrainingLoss(bpr_weight=args.bpr_weight)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_bpr = 0.0

        for batch in train_loader:
            # Positive samples
            pos_scores = model(
                batch["text_feat"].to(device),
                batch["visual_feat"].to(device),
                batch["interaction_feat"].to(device),
                batch["user_feat"].to(device),
                batch["context_feat"].to(device),
            )

            # Negative samples (in-batch negatives from shuffled features)
            neg_scores = model(
                batch["neg_text_feat"].to(device),
                batch["neg_visual_feat"].to(device),
                batch["neg_interaction_feat"].to(device),
                batch["user_feat"].to(device),  # same user
                batch["context_feat"].to(device),
            )

            pos_labels = batch["label"].to(device)
            neg_labels = 1.0 - pos_labels

            losses = criterion(pos_scores, neg_scores, pos_labels, neg_labels)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += losses["total"].item()
            total_bpr += losses["bpr"].item()

        n = len(train_loader)
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {total_loss/n:.4f} | "
            f"BPR: {total_bpr/n:.4f}"
        )

    torch.save(model.state_dict(), os.path.join(args.output_dir, "sarm_model.pt"))
    print(f"\nModel saved to {args.output_dir}/sarm_model.pt")


def main():
    parser = argparse.ArgumentParser(description="Train SARM model")
    parser.add_argument("--num_samples", type=int, default=2000, help="Toy dataset size")
    parser.add_argument("--anchor_tokens", type=int, default=8, help="K anchor tokens per stream")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_lr", type=float, default=1e-4)
    parser.add_argument("--anchor_lr", type=float, default=5e-4, help="Higher LR for anchors")
    parser.add_argument("--bpr_weight", type=float, default=0.5)
    parser.add_argument("--output_dir", default="checkpoints/")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
