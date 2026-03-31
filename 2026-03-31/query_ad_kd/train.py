from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import make_pairs, split
from model import StudentTwoTower, Teacher, distill_loss


def build_ad_graph(ad_embs: torch.Tensor, topk: int = 6) -> torch.Tensor:
    """Build a kNN adjacency matrix by cosine similarity."""
    embs = F.normalize(ad_embs, dim=-1)
    sim = embs @ embs.t()
    sim.fill_diagonal_(-1e9)
    _, idx = torch.topk(sim, k=min(topk, sim.shape[0] - 1), dim=-1)
    adj = torch.zeros_like(sim)
    adj.scatter_(1, idx, 1.0)
    adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adj


def label_propagation(seed_labels: torch.Tensor, adj: torch.Tensor, steps: int = 6) -> torch.Tensor:
    y = seed_labels.float()
    for _ in range(steps):
        y = 0.6 * y + 0.4 * (adj @ y)
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs_teacher", type=int, default=5)
    ap.add_argument("--epochs_student", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    pairs, ad_embs = make_pairs(seed=0)
    tr, _ = split(pairs, 0.9)

    # Small human-labeled set (simulate limited labels)
    labeled = [ex for ex in tr[:12000] if ex.y == 1][:120] + tr[:900]

    teacher = Teacher()
    opt_t = torch.optim.AdamW(teacher.parameters(), lr=args.lr)

    for ep in range(args.epochs_teacher):
        teacher.train()
        total = 0.0
        n = 0
        for ex in labeled:
            logit = teacher(ex.q, ex.ad_title, ex.ad_img)
            loss = F.binary_cross_entropy_with_logits(logit.unsqueeze(0), torch.tensor([float(ex.y)]))
            opt_t.zero_grad(set_to_none=True)
            loss.backward()
            opt_t.step()
            total += float(loss.detach())
            n += 1
        print(f"teacher epoch={ep} loss={total/max(1,n):.4f}")

    # Graph mining: propagate rare positives on ad graph to create more candidates
    adj = build_ad_graph(ad_embs)
    seed_labels = torch.zeros((ad_embs.shape[0],), dtype=torch.float32)
    for ex in labeled:
        if ex.y == 1:
            seed_labels[ex.ad_id] = 1.0
    scores = label_propagation(seed_labels, adj)

    # Teacher pseudo labels on a mined pool
    mined_ads = torch.argsort(scores, descending=True)[:800].tolist()
    pseudo = [ex for ex in tr[:30000] if ex.ad_id in set(mined_ads)]

    teacher.eval()
    with torch.no_grad():
        teacher_logits = torch.tensor([float(teacher(ex.q, ex.ad_title, ex.ad_img).item()) for ex in pseudo])

    student = StudentTwoTower()
    opt_s = torch.optim.AdamW(student.parameters(), lr=args.lr)

    for ep in range(args.epochs_student):
        student.train()
        total = 0.0
        n = 0
        for i, ex in enumerate(pseudo):
            slog = student(ex.q, ex.ad_title, ex.ad_img)
            y = torch.tensor(float(ex.y))
            loss = distill_loss(slog, teacher_logits[i], y)
            opt_s.zero_grad(set_to_none=True)
            loss.backward()
            opt_s.step()
            total += float(loss.detach())
            n += 1
        print(f"student epoch={ep} loss={total/max(1,n):.4f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"teacher": teacher.state_dict(), "student": student.state_dict()}, ckpt / "query_ad_kd.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
