"""
Training for CQ-SID + EG-GRPO.

Two-stage training:
  Stage 1: Train CQ-SID encoder with category-aware + query-item contrastive losses
  Stage 2: Train generative retriever with EG-GRPO using expert ranker reward
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import get_dataloaders
from cq_sid import CQSIDEncoder
from eg_grpo import GenerativeRetriever, ExpertRanker, eg_grpo_loss


def train_cq_sid(encoder, loader, optimizer, device, n_epochs=10):
    """Stage 1: Train CQ-SID with contrastive + VQ losses."""
    encoder.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        n = 0
        for batch in loader:
            pf = batch["product_feat"].to(device)
            pc = batch["product_cat"].to(device)
            qf = batch["query_feat"].to(device)

            codes, vq_loss, cat_loss, qi_loss, z_p = encoder(pf, pc, qf)

            loss = vq_loss + 0.5 * qi_loss + 0.3 * cat_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        print(f"  Stage1 Epoch {epoch+1}/{n_epochs} | loss={total_loss/n:.4f}")

    return encoder


def train_eg_grpo(retriever, expert, encoder, loader, optimizer, device, n_epochs=5):
    """Stage 2: Train generative retriever with EG-GRPO."""
    retriever.train()
    encoder.eval()

    for epoch in range(n_epochs):
        total_loss = 0.0
        n = 0
        for batch in loader:
            qf = batch["query_feat"].to(device)
            pf = batch["product_feat"].to(device)
            pc = batch["product_cat"].to(device)

            # Get target CQ-SID codes from encoder (no grad)
            with torch.no_grad():
                codes, _, _, _, _ = encoder(pf, pc)

            # EG-GRPO: sample trajectories, compute expert reward, optimize
            grpo_loss, mean_reward = eg_grpo_loss(
                retriever, expert, qf, n_samples=4, temperature=1.0
            )

            # Supervised learning on target codes (combines with GRPO)
            logits_all, _ = retriever(qf, target_codes=codes)
            sl_loss = sum(
                nn.CrossEntropyLoss()(logits_all[k], codes[:, k])
                for k in range(len(logits_all))
            )

            loss = grpo_loss + 0.5 * sl_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        print(f"  Stage2 Epoch {epoch+1}/{n_epochs} | loss={total_loss/n:.4f} | mean_reward={mean_reward:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, data = get_dataloaders(
        batch_size=256, n_products=1000, n_categories=30, n_queries=5000, n_interactions=30000,
        feat_dim=128
    )

    encoder = CQSIDEncoder(
        feat_dim=128, n_categories=30, cat_embed_dim=32,
        hidden_dim=128, n_levels=4, codebook_size=256
    ).to(device)

    retriever = GenerativeRetriever(
        query_dim=128, hidden_dim=128, n_levels=4, codebook_size=256
    ).to(device)

    expert = ExpertRanker(
        query_dim=128, hidden_dim=64, n_levels=4, codebook_size=256
    ).to(device)

    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Retriever params: {sum(p.numel() for p in retriever.parameters()):,}")

    # Stage 1: CQ-SID encoder training
    print("\n=== Stage 1: CQ-SID Encoder Training ===")
    opt1 = optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    train_cq_sid(encoder, train_loader, opt1, device, n_epochs=8)

    # Stage 2: EG-GRPO retriever training
    print("\n=== Stage 2: EG-GRPO Retriever Training ===")
    opt2 = optim.AdamW(retriever.parameters(), lr=5e-4, weight_decay=1e-4)
    train_eg_grpo(retriever, expert, encoder, train_loader, opt2, device, n_epochs=5)

    torch.save({"encoder": encoder.state_dict(), "retriever": retriever.state_dict()}, "cq_sid_checkpoint.pt")
    print("\nTraining complete. Checkpoint saved to cq_sid_checkpoint.pt")


if __name__ == "__main__":
    main()
