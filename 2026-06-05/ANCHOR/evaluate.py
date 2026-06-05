from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import Interaction, build_synthetic_dataset, split_interactions
from model import MFRecommender, NoiseRecognizer


def recall_ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, float]:
    # scores: [n_items], labels: {0,1}
    idx = np.argsort(-scores)[:k]
    hits = labels[idx]
    recall = float(hits.sum() / max(1.0, labels.sum()))

    dcg = 0.0
    for rank, rel in enumerate(hits, start=1):
        if rel <= 0:
            continue
        dcg += 1.0 / np.log2(rank + 1)

    # Ideal DCG
    ideal_hits = np.sort(labels)[::-1][:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal_hits, start=1):
        if rel <= 0:
            continue
        idcg += 1.0 / np.log2(rank + 1)

    ndcg = float(dcg / idcg) if idcg > 0 else 0.0
    return recall, ndcg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/anchor.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--noise-ratio", type=float, default=0.2)
    parser.add_argument("--eval-users", type=int, default=400)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    ds = build_synthetic_dataset(noise_ratio=args.noise_ratio, seed=args.seed)
    _, _, test = split_interactions(ds, seed=args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")

    recognizer = NoiseRecognizer(ds.n_users, ds.n_items, dim=ckpt["dim"]).to(device)
    recognizer.load_state_dict(ckpt["recognizer"])
    recognizer.eval()

    rec_model = MFRecommender(ds.n_users, ds.n_items, dim=ckpt["dim"]).to(device)
    rec_model.load_state_dict(ckpt["recommender"])
    rec_model.eval()

    # Build per-user positive sets from (clean) test interactions
    user_pos: dict[int, set[int]] = {u: set() for u in range(ds.n_users)}
    for it in test:
        if it.is_noise:
            continue
        user_pos[it.user_id].add(it.item_id)

    users = [u for u, items in user_pos.items() if len(items) >= 5]
    random.shuffle(users)
    users = users[: args.eval_users]

    recalls = []
    ndcgs = []

    all_items = torch.arange(ds.n_items, dtype=torch.long, device=device)

    with torch.no_grad():
        for u in users:
            u_ids = torch.full((ds.n_items,), u, dtype=torch.long, device=device)
            scores = rec_model(u_ids, all_items).detach().cpu().numpy()

            labels = np.zeros(ds.n_items, dtype=np.int64)
            for i in user_pos[u]:
                labels[i] = 1

            r, n = recall_ndcg_at_k(scores, labels, k=args.k)
            recalls.append(r)
            ndcgs.append(n)

    print(f"Recall@{args.k}={np.mean(recalls):.4f} NDCG@{args.k}={np.mean(ndcgs):.4f} users={len(users)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
