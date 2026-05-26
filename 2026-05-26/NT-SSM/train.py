from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data import ToyCFConfig, build_toy_cf_split
from graph import build_adjacency, compute_s_tilde, normalize_adjacency
from losses import compute_scores_for_batch, nt_ssm_loss, ssm_loss
from model import LightGCN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--loss", choices=["ssm", "nt_ssm"], default="nt_ssm")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--neg", type=int, default=32)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # NT-SSM coefficients (search range in paper: [0.5, 1.5])
    p.add_argument("--alpha_I_U", type=float, default=1.2)
    p.add_argument("--alpha_I_I", type=float, default=1.1)
    p.add_argument("--alpha_U_U", type=float, default=1.2)
    p.add_argument("--alpha_U_I", type=float, default=1.1)

    return p.parse_args()


def sample_neg_items(rng: np.random.Generator, *, num_items: int, avoid: set[int], k: int) -> np.ndarray:
    out: list[int] = []
    while len(out) < k:
        cand = int(rng.integers(0, num_items))
        if cand not in avoid:
            out.append(cand)
    return np.array(out, dtype=np.int64)


def sample_neg_users(rng: np.random.Generator, *, num_users: int, avoid: int, k: int) -> np.ndarray:
    out: list[int] = []
    while len(out) < k:
        cand = int(rng.integers(0, num_users))
        if cand != avoid:
            out.append(cand)
    return np.array(out, dtype=np.int64)


@torch.no_grad()
def eval_recall_ndcg(
    *,
    model: LightGCN,
    num_users: int,
    num_items: int,
    train_pos_by_user: dict[int, set[int]],
    gt_by_user: dict[int, set[int]],
    k: int = 20,
) -> dict[str, float]:
    model.eval()
    emb = model.forward_all()["e_all"]  # (U+I, D)

    recalls = []
    ndcgs = []

    for u in range(num_users):
        user_vec = emb[u]
        item_vecs = emb[num_users : num_users + num_items]
        scores = (item_vecs * user_vec.unsqueeze(0)).sum(dim=1)  # (I,)

        # Mask training positives
        for i in train_pos_by_user[u]:
            scores[i] = -1e9

        topk = torch.topk(scores, k=k).indices.cpu().numpy().tolist()
        gt = gt_by_user[u]

        hit = [1 if i in gt else 0 for i in topk]
        recall = sum(hit) / max(1, len(gt))

        # NDCG@k (single gt item in our toy split)
        dcg = 0.0
        for rank, rel in enumerate(hit, start=1):
            if rel:
                dcg += 1.0 / np.log2(rank + 1)
        idcg = 1.0  # with one relevant item
        ndcg = dcg / idcg

        recalls.append(recall)
        ndcgs.append(ndcg)

    return {"recall@20": float(np.mean(recalls)), "ndcg@20": float(np.mean(ndcgs))}


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    cfg = ToyCFConfig(seed=args.seed)
    split = build_toy_cf_split(cfg)

    # Build graph from training edges only (standard CF setting)
    a = build_adjacency(num_users=cfg.num_users, num_items=cfg.num_items, edges=split.train_edges)
    a_norm = normalize_adjacency(a)
    s_tilde = compute_s_tilde(a_norm=a_norm, num_layers=args.layers)

    s_tilde_t = torch.from_numpy(s_tilde).to(args.device)

    model = LightGCN(num_nodes=cfg.num_users + cfg.num_items, dim=args.dim, s_tilde=s_tilde_t).to(args.device)
    model.set_type_masks(num_users=cfg.num_users)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / f"{args.loss}.pt"

    edges = split.train_edges

    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(edges)

        for start in range(0, len(edges), args.batch_size):
            batch_edges = edges[start : start + args.batch_size]
            if not batch_edges:
                continue

            u = np.array([x[0] for x in batch_edges], dtype=np.int64)
            i = np.array([x[1] for x in batch_edges], dtype=np.int64)

            neg_items = np.stack(
                [sample_neg_items(rng, num_items=cfg.num_items, avoid=split.train_pos_by_user[int(uu)], k=args.neg) for uu in u],
                axis=0,
            )
            neg_users = np.stack(
                [sample_neg_users(rng, num_users=cfg.num_users, avoid=int(uu), k=args.neg) for uu in u],
                axis=0,
            )

            user_nodes = torch.from_numpy(u).to(args.device)
            item_nodes = torch.from_numpy(cfg.num_users + i).to(args.device)
            neg_item_nodes = torch.from_numpy(cfg.num_users + neg_items).to(args.device)
            neg_user_nodes = torch.from_numpy(neg_users).to(args.device)

            emb = model.forward_all()
            scores = compute_scores_for_batch(
                emb=emb,
                user_nodes=user_nodes,
                item_nodes=item_nodes,
                neg_item_nodes=neg_item_nodes,
                neg_user_nodes=neg_user_nodes,
                alpha_I_U=args.alpha_I_U,
                alpha_I_I=args.alpha_I_I,
                alpha_U_U=args.alpha_U_U,
                alpha_U_I=args.alpha_U_I,
            )

            if args.loss == "ssm":
                loss = ssm_loss(pos_score=scores["pos"], neg_score=scores["ssm_neg_items"], tau=args.tau)
            else:
                loss = nt_ssm_loss(
                    pos_score=scores["pos"],
                    neg_item_score=scores["nt_neg_items"],
                    neg_user_score=scores["nt_neg_users"],
                    tau=args.tau,
                )

            opt.zero_grad()
            loss.backward()
            opt.step()

        val = eval_recall_ndcg(
            model=model,
            num_users=cfg.num_users,
            num_items=cfg.num_items,
            train_pos_by_user=split.train_pos_by_user,
            gt_by_user=split.val_pos_by_user,
        )
        test = eval_recall_ndcg(
            model=model,
            num_users=cfg.num_users,
            num_items=cfg.num_items,
            train_pos_by_user=split.train_pos_by_user,
            gt_by_user=split.test_pos_by_user,
        )

        print(
            f"epoch={epoch} loss={args.loss} val_ndcg@20={val['ndcg@20']:.4f} val_recall@20={val['recall@20']:.4f} test_ndcg@20={test['ndcg@20']:.4f}"
        )

        if val["ndcg@20"] > best_val:
            best_val = val["ndcg@20"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "cfg": cfg,
                    "s_tilde": s_tilde,
                    "train_pos_by_user": split.train_pos_by_user,
                    "test_pos_by_user": split.test_pos_by_user,
                },
                ckpt_path,
            )

    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()
