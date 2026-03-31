from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import QueryAdDataset, collate
from model import CrossEncoderTeacher, StudentTwoTower, kd_loss


@torch.no_grad()
def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)

    sum_pos_ranks = float(ranks[y_true == 1].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def evaluate(model: StudentTwoTower, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys = []
    ps = []
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        logit, _ = model(batch["q"], batch["ad"], batch["img"], batch["ctx"])
        prob = torch.sigmoid(logit)
        ys.extend(batch["y"].detach().cpu().tolist())
        ps.extend(prob.detach().cpu().tolist())

    y = np.asarray(ys)
    p = np.asarray(ps)
    auc = roc_auc_binary((y > 0.5).astype(np.int32), p)
    return {"auc": float(auc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--teacher_epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ckpt", type=str, default="checkpoints/query_ad_kd.pt")
    args = ap.parse_args()

    ds = QueryAdDataset(jsonl_path=args.data or None, seed=args.seed)
    n = len(ds)
    split = int(n * 0.9)
    train_ds = torch.utils.data.Subset(ds, list(range(split)))
    val_ds = torch.utils.data.Subset(ds, list(range(split, n)))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = CrossEncoderTeacher().to(device)
    student = StudentTwoTower().to(device)

    opt_t = torch.optim.AdamW(teacher.parameters(), lr=args.lr, weight_decay=0.01)
    opt_s = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)

    # ------------------ train teacher ------------------
    for epoch in range(1, args.teacher_epochs + 1):
        teacher.train()
        losses = []
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)

            t_logit, _ = teacher(batch["q"], batch["ad"], batch["img"], batch["ctx"])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(t_logit, batch["y"])
            opt_t.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            opt_t.step()
            losses.append(float(loss.detach().cpu()))

        print(f"teacher epoch={epoch} loss={np.mean(losses):.4f}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ------------------ distill student ------------------
    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        student.train()
        losses = []
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)

            with torch.no_grad():
                t_logit, t_rep = teacher(batch["q"], batch["ad"], batch["img"], batch["ctx"])

            s_logit, s_rep = student(batch["q"], batch["ad"], batch["img"], batch["ctx"])
            loss = kd_loss(
                student_logit=s_logit,
                teacher_logit=t_logit,
                y=batch["y"],
                student_rep=s_rep,
                teacher_rep=t_rep,
            )

            opt_s.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt_s.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate(student, val_loader, device)
        print(f"student epoch={epoch} loss={np.mean(losses):.4f} metrics={metrics}")

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
            torch.save(
                {
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "args": vars(args),
                },
                args.ckpt,
            )

    print(f"saved: {args.ckpt} best_auc={best_auc:.4f}")


if __name__ == "__main__":
    main()
