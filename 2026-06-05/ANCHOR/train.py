from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import Interaction, build_synthetic_dataset, split_interactions
from model import MFRecommender, NoiseRecognizer, bpr_loss


def batch_to_tensors(batch: list[Interaction], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = torch.tensor([x.user_id for x in batch], dtype=torch.long, device=device)
    i = torch.tensor([x.item_id for x in batch], dtype=torch.long, device=device)
    y = torch.tensor([x.is_noise for x in batch], dtype=torch.long, device=device)
    return u, i, y


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--noise-ratio", type=float, default=0.2)
    parser.add_argument("--rec-epochs", type=int, default=3)
    parser.add_argument("--rec-lr", type=float, default=2e-3)
    parser.add_argument("--noise-epochs", type=int, default=3)
    parser.add_argument("--noise-lr", type=float, default=2e-3)
    parser.add_argument("--denoise-threshold", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="runs/anchor.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    ds = build_synthetic_dataset(noise_ratio=args.noise_ratio, seed=args.seed)
    train, val, test = split_interactions(ds, seed=args.seed)

    recognizer = NoiseRecognizer(ds.n_users, ds.n_items, dim=args.dim).to(device)
    rec_model = MFRecommender(ds.n_users, ds.n_items, dim=args.dim).to(device)

    # ----------------------------
    # Stage 1: train noise recognizer
    # ----------------------------
    opt = torch.optim.Adam(recognizer.parameters(), lr=args.noise_lr)
    ce = torch.nn.CrossEntropyLoss()

    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=lambda x: x)
    for epoch in range(args.noise_epochs):
        recognizer.train()
        losses = []
        for batch in loader:
            u, i, y = batch_to_tensors(batch, device)
            logits = recognizer(u, i)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        # quick val
        recognizer.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in DataLoader(val, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=lambda x: x):
                u, i, y = batch_to_tensors(batch, device)
                p = recognizer(u, i).argmax(dim=1)
                y_true.extend(y.detach().cpu().numpy().tolist())
                y_pred.extend(p.detach().cpu().numpy().tolist())
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall) / max(1e-12, precision + recall)
        print(f"[noise] epoch={epoch+1}/{args.noise_epochs} loss={np.mean(losses):.4f} val_f1={f1:.4f}")

    # ----------------------------
    # Stage 2: denoise interactions, then train recommender
    # ----------------------------
    recognizer.eval()
    kept_train: list[Interaction] = []
    with torch.no_grad():
        for batch in DataLoader(train, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=lambda x: x):
            u, i, _ = batch_to_tensors(batch, device)
            prob_noise = torch.softmax(recognizer(u, i), dim=1)[:, 1]
            keep = prob_noise < args.denoise_threshold
            keep_idx = keep.detach().cpu().numpy().tolist()
            for ex, k in zip(batch, keep_idx):
                if k:
                    kept_train.append(ex)

    print(f"[denoise] kept {len(kept_train)}/{len(train)} interactions")

    opt_rec = torch.optim.Adam(rec_model.parameters(), lr=args.rec_lr)

    # Pre-sample negatives for speed
    all_items = list(range(ds.n_items))

    for epoch in range(args.rec_epochs):
        rec_model.train()
        losses = []
        for batch in DataLoader(kept_train, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=lambda x: x):
            u = torch.tensor([x.user_id for x in batch], dtype=torch.long, device=device)
            pos_i = torch.tensor([x.item_id for x in batch], dtype=torch.long, device=device)
            neg_i = torch.tensor([random.choice(all_items) for _ in batch], dtype=torch.long, device=device)

            pos_s = rec_model(u, pos_i)
            neg_s = rec_model(u, neg_i)
            loss = bpr_loss(pos_s, neg_s)
            opt_rec.zero_grad(set_to_none=True)
            loss.backward()
            opt_rec.step()
            losses.append(float(loss.item()))

        print(f"[rec] epoch={epoch+1}/{args.rec_epochs} loss={np.mean(losses):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "recognizer": recognizer.state_dict(),
            "recommender": rec_model.state_dict(),
            "n_users": ds.n_users,
            "n_items": ds.n_items,
            "dim": args.dim,
        },
        args.out,
    )
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
