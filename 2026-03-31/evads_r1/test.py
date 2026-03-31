from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from dataset import TASKS, make_dataset, split
from model import EVAdsPolicy


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Classic Average Precision (area under PR curve) implementation.
    order = np.argsort(-y_score)
    y_true = y_true[order]
    cum_tp = np.cumsum(y_true)
    precision = cum_tp / (np.arange(len(y_true)) + 1)
    denom = max(1, int(y_true.sum()))
    return float((precision * y_true).sum() / denom)


def main() -> None:
    data = make_dataset(n=2000, seed=7)
    _, te = split(data, 0.0)

    model = EVAdsPolicy(vocab=1000, d_v=48, d=128, n_classes=12)
    ckpt = Path("checkpoints/evads_r1.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=False)
    model.eval()

    correct = 0
    total = 0

    y_true = []
    y_score = []

    by_task = {t: [0, 0] for t in TASKS}

    for ex in te:
        logits = model(ex.video, ex.asr, ex.ocr, ex.q)
        pred = int(torch.argmax(logits).item())

        total += 1
        correct += int(pred == ex.y)
        by_task[ex.task][0] += int(pred == ex.y)
        by_task[ex.task][1] += 1

        # Treat class 0 as the (rare) positive class for AUCPR demo.
        probs = torch.softmax(logits, dim=-1)
        y_true.append(1 if ex.y == 0 else 0)
        y_score.append(float(probs[0].detach()))

    acc = correct / max(1, total)
    aucpr = average_precision(np.array(y_true, dtype=np.int64), np.array(y_score, dtype=np.float64))

    print(f"Overall accuracy: {acc:.3f}")
    print(f"AUCPR (class 0 as positive): {aucpr:.6f}")
    print("By task accuracy:")
    for t in TASKS:
        c, n = by_task[t]
        print(f"  {t}: {c/max(1,n):.3f} (n={n})")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
