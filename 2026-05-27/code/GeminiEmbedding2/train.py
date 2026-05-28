"""
Training script for Gemini Embedding 2 reproduction.

Multi-stage training following the paper:
  Stage 1: Web-scale text-text contrastive (broad coverage)
  Stage 2: Domain-specific multi-task fine-tuning

Usage:
    python train.py --stage 1 --epochs 5 --batch_size 32 --lr 1e-4
    python train.py --stage 2 --epochs 3 --batch_size 16 --lr 5e-5
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import GE2Config, GeminiEmbeddingModel, MultiTaskInfoNCELoss
from dataset import get_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def compute_recall_at_k(q_embeds: torch.Tensor, k_embeds: torch.Tensor,
                        k: int = 1) -> float:
    """Compute Recall@K for a batch of query-key pairs."""
    sims = torch.matmul(q_embeds, k_embeds.T)
    ranks = sims.argsort(dim=-1, descending=True)
    labels = torch.arange(len(q_embeds), device=q_embeds.device)
    top_k = ranks[:, :k]
    hits = (top_k == labels.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()


def train_one_epoch(model: GeminiEmbeddingModel,
                    criterion: MultiTaskInfoNCELoss,
                    loader, optimizer, device: torch.device,
                    stage: int) -> float:
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        # Encode query and key text tokens
        q_ids = batch["query_ids"].to(device)
        q_mask = batch["query_mask"].to(device)
        k_ids = batch["key_ids"].to(device)
        k_mask = batch["key_mask"].to(device)
        task_ids = batch["task_id"].to(device)

        # Encode tokens (text path)
        q_tokens = model.encode_text(q_ids)
        k_tokens = model.encode_text(k_ids)

        # Forward through backbone
        q_embed = model(q_tokens, q_mask)
        k_embed = model(k_tokens, k_mask)

        loss = criterion({"query": q_embed, "key": k_embed}, task_ids)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        if step % 20 == 0:
            log.info(f"  step {step:4d} | loss {loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model: GeminiEmbeddingModel, loader, device: torch.device) -> dict:
    model.eval()
    all_q, all_k = [], []
    for batch in loader:
        q_ids = batch["query_ids"].to(device)
        q_mask = batch["query_mask"].to(device)
        k_ids = batch["key_ids"].to(device)
        k_mask = batch["key_mask"].to(device)
        q_tokens = model.encode_text(q_ids)
        k_tokens = model.encode_text(k_ids)
        all_q.append(model(q_tokens, q_mask))
        all_k.append(model(k_tokens, k_mask))
    q_embeds = torch.cat(all_q)
    k_embeds = torch.cat(all_k)
    return {
        "R@1": compute_recall_at_k(q_embeds, k_embeds, k=1),
        "R@5": compute_recall_at_k(q_embeds, k_embeds, k=5),
        "R@10": compute_recall_at_k(q_embeds, k_embeds, k=10),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256,  # toy scale
                        help="Hidden dim (use 768+ for real training)")
    parser.add_argument("--num_layers", type=int, default=4,    # toy scale
                        help="Number of transformer layers")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Toy-scale config (increase for real training)
    cfg = GE2Config(
        hidden_dim=args.hidden_dim,
        num_heads=max(1, args.hidden_dim // 64),
        num_layers=args.num_layers,
        ffn_dim=args.hidden_dim * 4,
        embed_dim=args.embed_dim,
    )

    model = GeminiEmbeddingModel(cfg).to(device)
    criterion = MultiTaskInfoNCELoss(temperature=0.02, num_tasks=4).to(device)
    optimizer = AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    train_loader, val_loader = get_dataloaders(cfg, batch_size=args.batch_size)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    os.makedirs(args.output_dir, exist_ok=True)
    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"Stage {args.stage} training for {args.epochs} epochs")

    for epoch in range(1, args.epochs + 1):
        log.info(f"=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(model, criterion, train_loader,
                                     optimizer, device, stage=args.stage)
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        log.info(f"Epoch {epoch} | train_loss={train_loss:.4f} | "
                 f"R@1={metrics['R@1']:.4f} | R@5={metrics['R@5']:.4f} | "
                 f"R@10={metrics['R@10']:.4f}")

    ckpt_path = os.path.join(args.output_dir, f"stage{args.stage}_final.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
    log.info(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
