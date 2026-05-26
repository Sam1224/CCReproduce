from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader

from data import ToyAdsConfig, ToyAdsDataset, set_seed
from model import CTRModel, aggregate_history_lastn, aggregate_history_mmr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["lastn", "memento_repr", "memento_data"], default="memento_repr")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--history-len", type=int, default=200)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--lambda-relevance", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Data memento (replay)
    p.add_argument("--replay-k", type=int, default=8)
    p.add_argument("--replay-weight", type=float, default=0.3)

    return p.parse_args()


@torch.no_grad()
def eval_split(model: CTRModel, loader: DataLoader, *, mode: str, k: int, lambda_relevance: float, device: str):
    model.eval()

    all_pred: list[float] = []
    all_y: list[float] = []

    for batch in loader:
        query_vec = batch["query_vec"].to(device)
        history_vecs = batch["history_vecs"].to(device)
        y = batch["label"].to(device)

        if mode == "lastn":
            hist_agg = aggregate_history_lastn(history_vecs, k=k)
        else:
            hist_agg = aggregate_history_mmr(
                history_vecs=history_vecs,
                query_vec=query_vec,
                k=k,
                lambda_relevance=lambda_relevance,
            )

        logits = model(query_vec, hist_agg)
        prob = torch.sigmoid(logits)

        all_pred.extend(prob.detach().cpu().numpy().tolist())
        all_y.extend(y.detach().cpu().numpy().tolist())

    auc = roc_auc_score(all_y, all_pred)
    ll = log_loss(all_y, all_pred, eps=1e-7)
    return {"auc": float(auc), "logloss": float(ll)}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = ToyAdsConfig(dim=args.dim, history_len=args.history_len, seed=args.seed)
    train_ds = ToyAdsDataset(cfg, split="train")
    val_ds = ToyAdsDataset(cfg, split="val")
    test_ds = ToyAdsDataset(cfg, split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = CTRModel(dim=args.dim).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    bce = torch.nn.BCEWithLogitsLoss()

    # A tiny replay buffer (Data Memento): store past (query_vec, history_vecs, label).
    replay_q: list[np.ndarray] = []
    replay_h: list[np.ndarray] = []
    replay_y: list[float] = []

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / f"{args.mode}.pt"

    def make_hist_agg(query_vec: torch.Tensor, history_vecs: torch.Tensor) -> torch.Tensor:
        if args.mode == "lastn":
            return aggregate_history_lastn(history_vecs, k=args.k)
        return aggregate_history_mmr(
            history_vecs=history_vecs,
            query_vec=query_vec,
            k=args.k,
            lambda_relevance=args.lambda_relevance,
        )

    best_val_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            query_vec = batch["query_vec"].to(args.device)
            history_vecs = batch["history_vecs"].to(args.device)
            y = batch["label"].to(args.device)

            hist_agg = make_hist_agg(query_vec, history_vecs)
            logits = model(query_vec, hist_agg)
            loss = bce(logits, y)

            # Data memento replay: retrieve similar past examples (cosine NN) and replay.
            if args.mode == "memento_data" and len(replay_q) >= args.replay_k:
                q_np = query_vec.detach().cpu().numpy()
                bank = np.stack(replay_q, axis=0)
                bank_norm = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12)
                q_norm = q_np / (np.linalg.norm(q_np, axis=1, keepdims=True) + 1e-12)
                sim = q_norm @ bank_norm.T  # (B, M)

                # top-k replay per row
                topk = np.argpartition(-sim, kth=args.replay_k - 1, axis=1)[:, : args.replay_k]

                replay_loss = 0.0
                replay_cnt = 0
                for b in range(q_np.shape[0]):
                    for j in topk[b]:
                        rq = torch.from_numpy(replay_q[j]).to(args.device)
                        rh = torch.from_numpy(replay_h[j]).to(args.device)
                        ry = torch.tensor(replay_y[j]).to(args.device)
                        r_hist_agg = make_hist_agg(rq.unsqueeze(0), rh.unsqueeze(0))
                        r_logit = model(rq.unsqueeze(0), r_hist_agg).squeeze(0)
                        replay_loss = replay_loss + bce(r_logit, ry)
                        replay_cnt += 1
                if replay_cnt > 0:
                    loss = loss + args.replay_weight * (replay_loss / replay_cnt)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # Update replay buffer with current batch (store CPU numpy)
            if args.mode == "memento_data":
                q_np = query_vec.detach().cpu().numpy()
                h_np = history_vecs.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                for b in range(q_np.shape[0]):
                    replay_q.append(q_np[b].copy())
                    replay_h.append(h_np[b].copy())
                    replay_y.append(float(y_np[b]))

                # Keep buffer bounded
                if len(replay_q) > 5000:
                    replay_q = replay_q[-5000:]
                    replay_h = replay_h[-5000:]
                    replay_y = replay_y[-5000:]

        val = eval_split(model, val_loader, mode=args.mode, k=args.k, lambda_relevance=args.lambda_relevance, device=args.device)
        test = eval_split(model, test_loader, mode=args.mode, k=args.k, lambda_relevance=args.lambda_relevance, device=args.device)

        print(f"epoch={epoch} val_auc={val['auc']:.4f} val_logloss={val['logloss']:.4f} test_auc={test['auc']:.4f}")

        if val["auc"] > best_val_auc:
            best_val_auc = val["auc"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)

    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()
