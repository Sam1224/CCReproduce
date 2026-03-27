from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyRankingDataset, collate_ranking
from model import OneModel


@torch.no_grad()
def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    # Very small toy AUC implementation.
    scores = scores.cpu().numpy().tolist()
    labels = labels.cpu().numpy().tolist()
    pos = [(s, l) for s, l in zip(scores, labels) if l == 1.0]
    neg = [(s, l) for s, l in zip(scores, labels) if l == 0.0]
    if not pos or not neg:
        return 0.5
    wins = 0
    for s_pos, _ in pos:
        for s_neg, _ in neg:
            wins += 1 if s_pos > s_neg else 0
    return wins / (len(pos) * len(neg))


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ToyRankingDataset(num_samples=256)
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_ranking)

    model = OneModel().to(device)
    model.eval()

    all_scores = []
    all_labels = []

    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        all_scores.append(out.pos_score)
        all_labels.append(batch["label"])

    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(f"Toy AUC (pos_score vs label): {_auc(scores, labels):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
