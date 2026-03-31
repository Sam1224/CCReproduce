from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ES3Builder, UniScaleDataset, collate_batch, iter_request_groups
from model import HHSFT, bce_loss, pairwise_ranking_loss


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    gains = labels[order]
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(labels)[::-1][:k]
    idcg = float((ideal * discounts[: ideal.size]).sum())
    return dcg / (idcg + 1e-9)


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
def evaluate(model: HHSFT, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_logits: List[float] = []
    all_labels: List[float] = []
    all_req: List[int] = []

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        logits = model(
            user_dense=batch["user_dense"],
            ctx_dense=batch["ctx_dense"],
            item_dense=batch["item_dense"],
            user_interest_dense=batch["user_interest_dense"],
            domain_id=batch["domain_id"],
        )
        all_logits.extend(logits.detach().cpu().tolist())
        all_labels.extend(batch["label"].detach().cpu().tolist())
        all_req.extend(batch["request_id"].detach().cpu().tolist())

    y = np.asarray(all_labels)
    p = 1.0 / (1.0 + np.exp(-np.asarray(all_logits)))

    auc = roc_auc_binary((y > 0.5).astype(np.int32), p)

    # Request-level NDCG
    req2 = {}
    for prob, lab, rid in zip(p.tolist(), y.tolist(), all_req):
        req2.setdefault(rid, []).append((prob, lab))

    ndcg3 = []
    ndcg10 = []
    for rid, pairs in req2.items():
        if len(pairs) < 2:
            continue
        s = np.asarray([x[0] for x in pairs])
        l = np.asarray([1.0 if x[1] > 0.5 else 0.0 for x in pairs])
        if l.sum() == 0:
            continue
        ndcg3.append(ndcg_at_k(s, l, 3))
        ndcg10.append(ndcg_at_k(s, l, 10))

    return {
        "auc": float(auc),
        "ndcg@3": float(np.mean(ndcg3) if ndcg3 else 0.0),
        "ndcg@10": float(np.mean(ndcg10) if ndcg10 else 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--requests", type=int, default=4000)
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ckpt", type=str, default="checkpoints/uniscale.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    builder = ES3Builder(seed=args.seed)
    samples = builder.build_requests(n_requests=args.requests, candidates=args.candidates)

    split = int(len(samples) * 0.9)
    train_ds = UniScaleDataset(samples[:split])
    val_ds = UniScaleDataset(samples[split:])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HHSFT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)

            logits = model(
                user_dense=batch["user_dense"],
                ctx_dense=batch["ctx_dense"],
                item_dense=batch["item_dense"],
                user_interest_dense=batch["user_interest_dense"],
                domain_id=batch["domain_id"],
            )

            labels = batch["label"].squeeze(-1)
            loss = bce_loss(logits, labels)

            # Request-level ranking loss (intra-request pairs).
            pr = []
            for g in iter_request_groups(batch["request_id"]):
                pr.append(pairwise_ranking_loss(logits, labels, g.to(device)))
            if pr:
                loss = loss + 0.2 * torch.stack(pr).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))

        metrics = evaluate(model, val_loader, device)
        print(f"epoch={epoch} loss={np.mean(losses):.4f} metrics={metrics}")

    import os

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.ckpt)
    print(f"saved: {args.ckpt}")


if __name__ == "__main__":
    main()
