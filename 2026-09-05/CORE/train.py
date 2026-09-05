from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import CompositionalListDataset, collate_samples, split_by_group
from model import COREConfig, COREEmbeddingModel, COREReranker, contrastive_level_loss, rank_kl_loss
from test import evaluate_rankings


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_teacher(reranker: COREReranker, loader: DataLoader, device: torch.device, epochs: int) -> None:
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=2e-4, weight_decay=0.01)
    loss_fn = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        reranker.train()
        total_loss = 0.0
        total = 0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            scores = reranker(batch["query_tokens"], batch["candidate_tokens"], batch["image_features"])
            loss = loss_fn(scores, batch["teacher_score"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * batch["level"].size(0)
            total += batch["level"].size(0)
        print(f"teacher epoch {epoch}: mse={total_loss / max(total, 1):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--teacher_epochs", type=int, default=2)
    parser.add_argument("--objective", choices=["rank_kl", "contrastive"], default="rank_kl")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_groups", type=int, default=512)
    parser.add_argument("--output", default="outputs/core_embed.pt")
    args = parser.parse_args()

    dataset = CompositionalListDataset(num_groups=args.num_groups)
    train_idx, test_idx = split_by_group(dataset)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate_samples)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = COREConfig(vocab_size=len(dataset.vocab), image_dim=dataset.image_dim)
    teacher = COREReranker(cfg).to(device)
    model = COREEmbeddingModel(cfg).to(device)

    train_teacher(teacher, train_loader, device, epochs=args.teacher_epochs)
    teacher.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            student_scores = model(batch["query_tokens"], batch["candidate_tokens"], batch["image_features"])
            if args.objective == "rank_kl":
                with torch.no_grad():
                    teacher_scores = teacher(batch["query_tokens"], batch["candidate_tokens"], batch["image_features"])
                loss = rank_kl_loss(student_scores, teacher_scores, batch["group_id"])
            else:
                loss = contrastive_level_loss(student_scores, batch["level"], batch["group_id"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * batch["level"].size(0)
            total += batch["level"].size(0)

        metrics = evaluate_rankings(model, test_loader, device)
        print(
            f"student epoch {epoch}: loss={total_loss / max(total, 1):.4f} "
            f"r@1={metrics['recall_at_1']:.4f} ndcg@5={metrics['ndcg_at_5']:.4f} "
            f"pair_acc={metrics['pairwise_order_acc']:.4f}"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "vocab": dataset.vocab}, args.output)
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
