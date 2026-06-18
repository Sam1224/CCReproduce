"""
AIR (Atomic Intent Reasoning) — Evaluation Script
Computes Recall@K, NDCG@K on the validation set.
"""

import argparse
import math
import torch
import numpy as np

from data import generate_toy_data, build_dataloaders
from model import AIRRankingModel


def ndcg_at_k(rec_ids: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """NDCG@K for single positive per user."""
    hits = (rec_ids[:, :k] == labels.unsqueeze(-1))  # (B, k)
    positions = torch.arange(1, k + 1, device=rec_ids.device).float()
    dcg = (hits.float() / torch.log2(positions + 1)).sum(dim=-1)
    return dcg.mean().item()


def recall_at_k(rec_ids: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    return (rec_ids[:, :k] == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()


@torch.no_grad()
def full_eval(model, loader, device, all_item_embs, ks=(10, 20, 50)):
    model.eval()
    all_rec = []
    all_labels = []

    for batch in loader:
        content_seq = batch["content_seq"].to(device)
        commerce_seq = batch["commerce_seq"].to(device)
        labels = batch["label"].to(device)

        rec_ids = model.recommend(content_seq, commerce_seq, all_item_embs, top_k=max(ks))
        all_rec.append(rec_ids.cpu())
        all_labels.append(labels.cpu())

    all_rec = torch.cat(all_rec)
    all_labels = torch.cat(all_labels)

    results = {}
    for k in ks:
        results[f"Recall@{k}"] = recall_at_k(all_rec, all_labels, k)
        results[f"NDCG@{k}"] = ndcg_at_k(all_rec, all_labels, k)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="air_best.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = generate_toy_data(seed=args.seed)
    _, val_loader = build_dataloaders(data, batch_size=args.batch_size)

    atom_proto = torch.FloatTensor(data["atom_prototypes"])
    model = AIRRankingModel(
        n_content_items=data["n_content_items"],
        n_commerce_items=data["n_commerce_items"],
        atom_prototypes=atom_proto,
    ).to(device)

    try:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded checkpoint: {args.ckpt}")
    except FileNotFoundError:
        print("No checkpoint found; evaluating with random weights.")

    all_item_embs = model.item_encoder.item_emb.weight.detach().to(device)
    results = full_eval(model, val_loader, device, all_item_embs)

    print("\n=== AIR Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
