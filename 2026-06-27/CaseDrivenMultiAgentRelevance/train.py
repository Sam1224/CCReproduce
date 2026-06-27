from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    return p.parse_args()


def train(model, train_ds, epochs: int = 5, batch_size: int = 128, lr: float = 2e-3) -> None:
    model.train()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total = 0.0
        for batch in loader:
            out = model(batch["q_ids"], batch["d_ids"])

            y = batch["y"]
            y_retrieval = (y > 0).to(torch.float32)

            loss_coarse = F.cross_entropy(out["coarse_logits"], y)
            loss_retrieval = F.binary_cross_entropy_with_logits(out["retrieval_logit"], y_retrieval)

            # fine-score: encourage monotonicity with label
            y_fine = (y.to(torch.float32) / 3.0).clamp(0, 1)
            loss_fine = F.mse_loss(torch.sigmoid(out["fine_score"]), y_fine)

            loss = loss_coarse + 0.3 * loss_retrieval + 0.2 * loss_fine

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())

        print(f"epoch={epoch} loss={total/len(loader):.4f}")
