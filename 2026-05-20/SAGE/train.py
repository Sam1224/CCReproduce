from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sage.dataset import make_toy_pu_dataset
from sage.harvest import harvest_confident_negatives, head_bucket_sample
from sage.metrics import compute_binary_metrics
from sage.model import MLPClassifier


def _train_one(
    *,
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    edge_val: torch.Tensor,
    out_path: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)

    x_train = torch.cat([x_pos, x_neg], dim=0)
    y_train = torch.cat([
        torch.ones((x_pos.shape[0],), dtype=torch.float32),
        torch.zeros((x_neg.shape[0],), dtype=torch.float32),
    ])

    ds = TensorDataset(x_train, y_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = MLPClassifier(n_features=x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _epoch in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_val.to(device)).cpu()
        y_pred = (torch.sigmoid(logits) >= 0.5).to(torch.int64)

    metrics = compute_binary_metrics(y_true=y_val, y_pred=y_pred, edge_cohort=edge_val)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_features": x_train.shape[1]}, out_path)

    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "edge_fpr": metrics.edge_fpr,
        "n_pos": int(x_pos.shape[0]),
        "n_neg": int(x_neg.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="runs/toy")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Harvesting hyperparams (toy defaults).
    parser.add_argument("--candidate_size", type=int, default=8000)
    parser.add_argument("--n_neg", type=int, default=2000)
    parser.add_argument("--floor_per_bucket", type=int, default=10)
    parser.add_argument("--maha_q", type=float, default=0.95)
    parser.add_argument("--knn_q", type=float, default=0.6)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = make_toy_pu_dataset(seed=args.seed)
    n = data.x.shape[0]

    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=rng)
    n_train = int(n * 0.7)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    labeled_pos_train = data.labeled_positive[train_idx]
    pos_idx = train_idx[labeled_pos_train]
    unlabeled_idx = train_idx[~labeled_pos_train]

    x_pos = data.x[pos_idx].float()
    x_unlabeled = data.x[unlabeled_idx].float()

    harvested_rel, stats = harvest_confident_negatives(
        x_unlabeled,
        candidate_size=args.candidate_size,
        floor_per_bucket=args.floor_per_bucket,
        target_harvest_size=args.n_neg,
        maha_quantile=args.maha_q,
        knn_quantile=args.knn_q,
        seed=args.seed,
    )
    x_neg_sage = x_unlabeled[harvested_rel]

    # Baseline: head-only negatives (representation bias: long-tail buckets are under-sampled).
    head_neg_rel = head_bucket_sample(x_unlabeled, n_samples=args.n_neg, head_coverage=0.8, seed=args.seed)
    x_neg_head = x_unlabeled[head_neg_rel]

    x_val = data.x[val_idx].float()
    y_val = data.y_true[val_idx]
    edge_val = data.edge_cohort[val_idx]

    res_sage = _train_one(
        x_pos=x_pos,
        x_neg=x_neg_sage,
        x_val=x_val,
        y_val=y_val,
        edge_val=edge_val,
        out_path=out_dir / "model_sage.pt",
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    res_head = _train_one(
        x_pos=x_pos,
        x_neg=x_neg_head,
        x_val=x_val,
        y_val=y_val,
        edge_val=edge_val,
        out_path=out_dir / "model_head_only.pt",
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    report = {
        "harvest_stats": {
            "n_unlabeled": stats.n_unlabeled,
            "n_candidates": stats.n_candidates,
            "n_harvested": stats.n_harvested,
            "pass_maha": stats.pass_maha,
            "pass_knn": stats.pass_knn,
        },
        "sage": res_sage,
        "head_only": res_head,
        "notes": "Toy reproduction. Absolute numbers are synthetic; focus is on pipeline alignment.",
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[SAGE]", json.dumps(res_sage, indent=2))
    print("[HeadOnly]", json.dumps(res_head, indent=2))
    print("[Harvest]", json.dumps(report["harvest_stats"], indent=2))


if __name__ == "__main__":
    main()
