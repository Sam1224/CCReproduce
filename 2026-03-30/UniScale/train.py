from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data.es3_sampler import PairSample, es3_expand
from data.toy_dataset import generate_toy_logs
from model.hhsft import HHSFTConfig, HHSFTModel
from scripts.metrics import auc_score, gauc_score, train_test_split


@dataclass(frozen=True)
class Batch:
    user_tokens: torch.Tensor
    query_tokens: torch.Tensor
    item_tokens: torch.Tensor
    domain_id: torch.Tensor
    click: torch.Tensor
    purchase: torch.Tensor
    query_id: torch.Tensor


class PairDataset(Dataset):
    def __init__(self, rows: List[PairSample]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> PairSample:
        return self.rows[idx]


def collate_fn(rows: List[PairSample]) -> Batch:
    user_tokens = torch.tensor(np.stack([r.user_tokens for r in rows]), dtype=torch.long)
    query_tokens = torch.tensor(np.stack([r.query_tokens for r in rows]), dtype=torch.long)
    item_tokens = torch.tensor(np.stack([r.item_tokens for r in rows]), dtype=torch.long)
    domain_id = torch.tensor([r.domain_id for r in rows], dtype=torch.long)
    click = torch.tensor([r.click for r in rows], dtype=torch.float32)
    purchase = torch.tensor([r.purchase for r in rows], dtype=torch.float32)
    query_id = torch.tensor([r.query_id for r in rows], dtype=torch.long)

    return Batch(
        user_tokens=user_tokens,
        query_tokens=query_tokens,
        item_tokens=item_tokens,
        domain_id=domain_id,
        click=click,
        purchase=purchase,
        query_id=query_id,
    )


@torch.no_grad()
def evaluate(model: HHSFTModel, loader: DataLoader, device: torch.device) -> None:
    model.eval()

    all_click: List[np.ndarray] = []
    all_click_pred: List[np.ndarray] = []
    all_qid: List[np.ndarray] = []

    for batch in loader:
        click_logit, _ = model(
            batch.user_tokens.to(device),
            batch.query_tokens.to(device),
            batch.item_tokens.to(device),
            batch.domain_id.to(device),
        )
        pred = torch.sigmoid(click_logit).cpu().numpy()

        all_click.append(batch.click.numpy())
        all_click_pred.append(pred)
        all_qid.append(batch.query_id.numpy())

    y = np.concatenate(all_click)
    s = np.concatenate(all_click_pred)
    g = np.concatenate(all_qid)

    print(f"AUC(click):  {auc_score(y, s):.4f}")
    print(f"GAUC(click): {gauc_score(g, y, s):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--use_es3", type=int, default=1)
    args = parser.parse_args()

    logs, item_vecs, query_vecs = generate_toy_logs(seed=7)

    if args.use_es3:
        rows = es3_expand(seed=7, logs=logs, item_vectors=item_vecs, query_vectors=query_vecs)
    else:
        rows = [
            PairSample(
                user_id=s.user_id,
                query_id=s.query_id,
                item_id=s.item_id,
                domain_id=s.domain_id,
                user_tokens=s.user_tokens,
                query_tokens=s.query_tokens,
                item_tokens=s.item_tokens,
                click=s.click,
                purchase=s.purchase,
                exposed=1,
            )
            for s in logs
        ]

    train_idx, test_idx = train_test_split(len(rows), test_ratio=0.2, seed=7)
    train_rows = [rows[int(i)] for i in train_idx]
    test_rows = [rows[int(i)] for i in test_idx]

    train_loader = DataLoader(PairDataset(train_rows), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(PairDataset(test_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = HHSFTConfig()
    model = HHSFTModel(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)

            click_logit, purchase_logit = model(
                batch.user_tokens.to(device),
                batch.query_tokens.to(device),
                batch.item_tokens.to(device),
                batch.domain_id.to(device),
            )

            click_loss = loss_fn(click_logit, batch.click.to(device))
            purchase_loss = loss_fn(purchase_logit, batch.purchase.to(device))

            # toy hierarchical attribution: purchase is a rarer, higher-level label
            loss = click_loss + 0.5 * purchase_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = batch.click.size(0)
            total_loss += float(loss.item()) * bs
            n += bs

        print(f"Epoch {epoch}: train loss {total_loss / max(1, n):.4f}")
        evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()
