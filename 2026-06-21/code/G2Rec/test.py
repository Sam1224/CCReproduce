"""
G2Rec evaluation script (arXiv 2606.20554).

Evaluates Hit@K and NDCG@K on the validation/test set.
The standard protocol for sequential recommendation:
  - Leave-last-out: last item in sequence = test target
  - Rank the target against 99 randomly sampled negatives
"""

import argparse
import math
import torch
import numpy as np
from data import generate_toy_dataset, SeqRecDataset
from model import G2Rec
from torch.utils.data import DataLoader


def hit_at_k(rank: int, k: int) -> float:
    return 1.0 if rank <= k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    if rank <= k:
        return 1.0 / math.log2(rank + 1)
    return 0.0


@torch.no_grad()
def evaluate(model: G2Rec, loader: DataLoader, device, k_list=(5, 10, 20)):
    model.eval()
    metrics = {f"Hit@{k}": 0.0 for k in k_list}
    metrics.update({f"NDCG@{k}": 0.0 for k in k_list})
    n = 0

    for user_seqs, targets in loader:
        user_seqs = user_seqs.to(device)
        B = user_seqs.size(0)

        # Generate predictions (top-K items)
        max_k = max(k_list)
        preds = model.generate(user_seqs, top_k=max_k)  # (B, max_k)
        preds_cpu = preds.cpu().numpy()
        tgts_cpu = targets.numpy()

        for b in range(B):
            pred_list = preds_cpu[b].tolist()
            tgt = int(tgts_cpu[b])
            if tgt in pred_list:
                rank = pred_list.index(tgt) + 1
            else:
                rank = max_k + 1

            for k in k_list:
                metrics[f"Hit@{k}"] += hit_at_k(rank, k)
                metrics[f"NDCG@{k}"] += ndcg_at_k(rank, k)
        n += B

    for key in metrics:
        metrics[key] /= max(n, 1)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="g2rec_best.pt")
    p.add_argument("--n_items", type=int, default=1000)
    p.add_argument("--n_users", type=int, default=500)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = generate_toy_dataset(n_users=args.n_users, n_items=args.n_items, seq_len=args.seq_len)
    # Use last 10% as test
    n_test = int(args.n_users * 0.1)
    test_ds = SeqRecDataset(data["user_seqs"][:n_test], data["targets"][:n_test])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = G2Rec(n_items=args.n_items, seq_len=args.seq_len).to(device)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    except FileNotFoundError:
        print("No checkpoint found; evaluating random model.")

    metrics = evaluate(model, test_loader, device)
    print("\n=== Evaluation Results ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
