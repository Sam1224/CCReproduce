"""
Evaluation script for FLUID: measures ranking quality with and without item IDs.

Compares:
1. ID-based baseline: uses room_id embedding directly
2. FLUID (ID-free): uses LUCID codes
"""

import torch
import torch.nn as nn
from data import get_dataloaders
from model import FLUIDRanker


class IDBasedBaseline(nn.Module):
    """Baseline that uses item IDs directly (no multimodal codes)."""

    def __init__(self, n_users: int, n_rooms: int, user_dim: int = 64, item_dim: int = 64):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, user_dim)
        self.item_embed = nn.Embedding(n_rooms, item_dim)
        self.scorer = nn.Sequential(
            nn.Linear(user_dim + item_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, user_ids, room_ids, **_):
        e_user = self.user_embed(user_ids)
        e_item = self.item_embed(room_ids)
        return self.scorer(torch.cat([e_user, e_item], -1)).squeeze(-1)


@torch.no_grad()
def compute_recall_at_k(model, loader, device, k: int = 10, is_id_based: bool = False):
    """Compute Recall@K: fraction of positive rooms in top-K per user."""
    model.eval()
    from collections import defaultdict
    user_scores = defaultdict(list)

    for batch in loader:
        user_ids = batch["user_id"].to(device)
        room_ids = batch["room_id"].to(device)
        visual = batch["visual_feat"].to(device)
        audio = batch["audio_feat"].to(device)
        text = batch["text_feat"].to(device)
        labels = batch["label"]

        if is_id_based:
            scores = model(user_ids, room_ids)
        else:
            scores, _ = model(user_ids, visual, audio, text)

        for uid, rid, sc, lb in zip(
            user_ids.cpu().tolist(), room_ids.cpu().tolist(),
            torch.sigmoid(scores).cpu().tolist(), labels.tolist()
        ):
            user_scores[uid].append((sc, lb))

    recall_vals = []
    for uid, score_label_pairs in user_scores.items():
        score_label_pairs.sort(key=lambda x: x[0], reverse=True)
        top_k = score_label_pairs[:k]
        n_pos_in_topk = sum(lb for _, lb in top_k)
        n_pos_total = sum(lb for _, lb in score_label_pairs)
        if n_pos_total > 0:
            recall_vals.append(n_pos_in_topk / n_pos_total)

    return sum(recall_vals) / len(recall_vals) if recall_vals else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_rooms, n_users = 500, 2000

    _, val_loader, data = get_dataloaders(
        batch_size=256, n_rooms=n_rooms, n_users=n_users, n_interactions=20000
    )

    # Load FLUID model
    fluid = FLUIDRanker(
        n_users=n_users, user_dim=64, visual_dim=256, audio_dim=128, text_dim=128,
        lucid_hidden_dim=128, n_levels=4, codebook_size=256, hidden_dim=64,
    ).to(device)
    try:
        fluid.load_state_dict(torch.load("fluid_checkpoint.pt", map_location=device))
        print("Loaded trained FLUID checkpoint.")
    except FileNotFoundError:
        print("No checkpoint found. Using untrained model for demo.")

    # ID-based baseline
    baseline = IDBasedBaseline(n_users=n_users, n_rooms=n_rooms).to(device)

    k = 10
    fluid_recall = compute_recall_at_k(fluid, val_loader, device, k=k, is_id_based=False)
    baseline_recall = compute_recall_at_k(baseline, val_loader, device, k=k, is_id_based=True)

    print(f"\n=== FLUID Evaluation (Recall@{k}) ===")
    print(f"ID-based Baseline:  {baseline_recall:.4f}")
    print(f"FLUID (ID-free):    {fluid_recall:.4f}")
    print(f"Improvement:        {(fluid_recall - baseline_recall):.4f}")
    print("\nNote: Toy data with random features; real gains require actual multimodal features.")


if __name__ == "__main__":
    main()
