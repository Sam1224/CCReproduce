from __future__ import annotations

"""Load a trained OneRank checkpoint and report per-task AUC (click / cart /
order) on a held-out toy split. AUC is computed with the rank-based
Mann-Whitney estimator (no sklearn dependency)."""

import argparse

import numpy as np
import torch

from data import ToyConfig, generate_dataset, iterate_batches
from model import OneRank, OneRankConfig, N_TASKS

TASK_NAMES = ["click", "cart", "order"]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U based AUC with average ranks for ties."""
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for tied scores
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for t in range(i, j + 1):
                ranks[order[t]] = avg
        i = j + 1
    sum_pos = ranks[pos].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="runs/onerank/ckpt.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt_path, map_location=device)

    toy = ToyConfig(**ckpt["toy_cfg"])
    mcfg = OneRankConfig(**ckpt["model_cfg"])
    model = OneRank(mcfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data = generate_dataset(toy, seed=ckpt["seed"])
    test = data["test"]

    all_scores = [[] for _ in range(N_TASKS)]
    all_labels = [[] for _ in range(N_TASKS)]
    with torch.no_grad():
        for batch in iterate_batches(test, args.batch_size, shuffle=False):
            hist = torch.as_tensor(batch["hist"], dtype=torch.long, device=device)
            anchor = torch.as_tensor(batch["anchor"], dtype=torch.long, device=device)
            cands = torch.as_tensor(batch["cands"], dtype=torch.long, device=device)
            labels = batch["labels"]                              # [B, N, 3]
            scores = model(hist, anchor, cands).cpu().numpy()     # [B, 3, N]
            for k in range(N_TASKS):
                all_scores[k].append(scores[:, k, :].reshape(-1))
                all_labels[k].append(labels[:, :, k].reshape(-1))

    print("=== OneRank per-task AUC (held-out toy split) ===")
    aucs = {}
    for k in range(N_TASKS):
        s = np.concatenate(all_scores[k])
        y = np.concatenate(all_labels[k])
        a = roc_auc(s, y)
        aucs[TASK_NAMES[k]] = a
        print(f"  {TASK_NAMES[k]:>6}-AUC = {a:.4f}   (pos rate = {y.mean():.3f})")
    print(f"  mean  -AUC = {np.nanmean(list(aucs.values())):.4f}")


if __name__ == "__main__":
    main()
