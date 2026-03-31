from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from dataset import make_pairs, split
from model import StudentTwoTower, Teacher


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    cum_tp = np.cumsum(y_true)
    precision = cum_tp / (np.arange(len(y_true)) + 1)
    denom = max(1, int(y_true.sum()))
    return float((precision * y_true).sum() / denom)


def main() -> None:
    pairs, _ = make_pairs(seed=11)
    _, te = split(pairs, 0.0)

    teacher = Teacher()
    student = StudentTwoTower()

    ckpt = Path("checkpoints/query_ad_kd.pt")
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        teacher.load_state_dict(state["teacher"], strict=False)
        student.load_state_dict(state["student"], strict=False)

    teacher.eval()
    student.eval()

    yt = np.array([ex.y for ex in te], dtype=np.int64)

    with torch.no_grad():
        t_score = np.array([float(torch.sigmoid(teacher(ex.q, ex.ad_title, ex.ad_img)).item()) for ex in te])
        s_score = np.array([float(torch.sigmoid(student(ex.q, ex.ad_title, ex.ad_img)).item()) for ex in te])

    aucpr_t = average_precision(yt, t_score)
    aucpr_s = average_precision(yt, s_score)

    print(f"Teacher AUCPR: {aucpr_t:.6f}")
    print(f"Student AUCPR: {aucpr_s:.6f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
