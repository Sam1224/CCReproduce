"""
Training script for SemanticNativeSFVRec.

Two-phase training (faithful to paper):
  Phase 1: Warm-up RQ-VAE — learn stable semantic codebooks from content embeddings
  Phase 2: End-to-end — jointly train sequence model + fine-tune RQ-VAE

Usage:
  python train.py --device cpu --warmup_epochs 2 --finetune_epochs 3
"""

import torch
import argparse
import torch.nn.functional as F
from model import SemanticNativeSFVRec, SemanticSFVConfig
from data import get_dataloader


def warmup_rqvae(model, dataloader, epochs: int = 2, lr: float = 1e-3, device: str = "cpu"):
    """Phase 1: Pre-train the RQ-VAE encoder to build stable semantic codebooks."""
    optimizer = torch.optim.AdamW(model.rqvae.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            content_embs = batch["target_content_emb"].to(device)  # [B, content_dim]

            _, _, vq_loss = model.rqvae(content_embs)

            optimizer.zero_grad()
            vq_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.rqvae.parameters(), 1.0)
            optimizer.step()
            total_loss += vq_loss.item()

            if step % 20 == 0:
                print(f"[RQ-VAE Warmup] Epoch {epoch+1} Step {step} vq_loss={vq_loss.item():.4f}")

        print(f"[RQ-VAE Warmup] Epoch {epoch+1} avg_loss={total_loss / max(len(dataloader), 1):.4f}")


def finetune_e2e(model, dataloader, epochs: int = 3, lr: float = 1e-4, device: str = "cpu"):
    """Phase 2: End-to-end fine-tuning — sequence model + RQ-VAE."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(dataloader)
    )
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_contrastive = 0.0

        for step, batch in enumerate(dataloader):
            hist_embs = batch["history_content_embs"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            tgt_emb = batch["target_content_emb"].to(device)

            out = model(hist_embs, attn_mask, tgt_emb)
            loss = out["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_contrastive += out["contrastive_loss"].item()

            if step % 20 == 0:
                print(
                    f"[E2E] Epoch {epoch+1} Step {step} "
                    f"loss={loss.item():.4f} "
                    f"contrastive={out['contrastive_loss'].item():.4f} "
                    f"vq={out['vq_loss'].item():.4f}"
                )

        n = max(len(dataloader), 1)
        print(
            f"[E2E] Epoch {epoch+1} avg_loss={total_loss/n:.4f} "
            f"avg_contrastive={total_contrastive/n:.4f}"
        )


@torch.no_grad()
def evaluate(model, dataloader, device: str = "cpu", top_k: int = 10):
    """
    Evaluate Recall@K and NDCG@K using in-batch negatives.
    Collects user representations and target embeddings from the full dataset,
    then computes approximate retrieval metrics.
    """
    model.eval()
    all_user_repr = []
    all_tgt_emb = []

    for batch in dataloader:
        hist_embs = batch["history_content_embs"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        tgt_emb = batch["target_content_emb"].to(device)

        user_repr = model.get_user_repr(hist_embs, attn_mask)
        item_emb = model.get_item_emb(tgt_emb)

        all_user_repr.append(user_repr.cpu())
        all_tgt_emb.append(item_emb.cpu())

    user_repr = torch.cat(all_user_repr, dim=0)  # [N, H]
    tgt_emb = torch.cat(all_tgt_emb, dim=0)      # [N, H]

    scores = user_repr @ tgt_emb.T  # [N, N]
    N = scores.size(0)
    labels = torch.arange(N)

    # Recall@K
    topk_ids = scores.topk(top_k, dim=-1).indices  # [N, K]
    hits = (topk_ids == labels.unsqueeze(-1)).any(dim=-1).float()
    recall_k = hits.mean().item()

    # NDCG@K
    rank = (topk_ids == labels.unsqueeze(-1)).float().argmax(dim=-1)
    hit_mask = hits.bool()
    ndcg_scores = torch.zeros(N)
    ndcg_scores[hit_mask] = 1.0 / torch.log2(rank[hit_mask].float() + 2)
    ndcg_k = ndcg_scores.mean().item()

    print(f"[Eval] Recall@{top_k}={recall_k:.4f}  NDCG@{top_k}={ndcg_k:.4f}")
    return {"recall": recall_k, "ndcg": ndcg_k}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--finetune_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    config = SemanticSFVConfig(
        content_dim=256,
        num_rq_levels=4,
        truncation_depth=2,
        codebook_size=512,
        hidden_size=256,
        num_heads=8,
        num_layers=4,
        max_seq_len=50,
    )

    device = args.device
    model = SemanticNativeSFVRec(config).to(device)
    dataloader = get_dataloader(batch_size=args.batch_size)

    print("=== Phase 1: RQ-VAE Warm-up ===")
    warmup_rqvae(model, dataloader, epochs=args.warmup_epochs, device=device)

    print("\n=== Phase 2: End-to-End Fine-tuning ===")
    finetune_e2e(model, dataloader, epochs=args.finetune_epochs, device=device)

    print("\n=== Evaluation ===")
    evaluate(model, dataloader, device=device, top_k=args.top_k)

    torch.save(model.state_dict(), "semantic_sfv_rec.pt")
    print("Saved model to semantic_sfv_rec.pt")


if __name__ == "__main__":
    main()
