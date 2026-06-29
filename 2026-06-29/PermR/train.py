from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data import SyntheticRerankDataset, collate_queries, split_dataset
from model import ScoringModel


def main() -> None:
    torch.manual_seed(0)

    dataset = SyntheticRerankDataset(num_queries=300, num_items=8, feature_dim=16, seed=42)
    train_idx, val_idx = split_dataset(dataset, train_ratio=0.85, seed=0)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=32,
        shuffle=True,
        collate_fn=collate_queries,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=64,
        shuffle=False,
        collate_fn=collate_queries,
        drop_last=False,
    )

    model = ScoringModel(feature_dim=16, hidden_dim=64)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    best_val = None
    best_state = None

    for epoch in range(1, 21):
        model.train()
        for batch in train_loader:
            x = batch["features"].float()
            rel = batch["rel"].float()
            revenue = batch["revenue"].float()

            rel_logit, rev_pred = model(x)
            loss = bce(rel_logit, rel) + 0.25 * mse(rev_pred, revenue)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            n = 0
            for batch in val_loader:
                x = batch["features"].float()
                rel = batch["rel"].float()
                revenue = batch["revenue"].float()

                rel_logit, rev_pred = model(x)
                loss = bce(rel_logit, rel) + 0.25 * mse(rev_pred, revenue)
                val_loss += float(loss.item())
                n += 1

        val_loss /= max(1, n)
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"epoch {epoch:02d}  val_loss={val_loss:.4f}")

    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / "scorer.pt"

    if best_state is None:
        best_state = model.state_dict()

    torch.save({"state_dict": best_state, "feature_dim": 16, "hidden_dim": 64}, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
