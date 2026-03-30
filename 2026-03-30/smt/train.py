from __future__ import annotations

import torch

from blocks import bce_loss
from dataset import make_toy_ranking_data, iter_batches
from model import SMTConfig, StandardModelTemplate


def to_torch(batch):
    return (
        torch.from_numpy(batch.cat).long(),
        torch.from_numpy(batch.x).float(),
        torch.from_numpy(batch.y).float(),
    )


def train_one(cfg: SMTConfig, steps: int = 300) -> float:
    tr, va = make_toy_ranking_data(n=5000, d=cfg.x_dim, n_cat=cfg.n_cat, seed=0)
    tr_it = iter_batches(*tr, batch_size=256, seed=0)
    va_it = iter_batches(*va, batch_size=512, seed=1)

    model = StandardModelTemplate(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(steps):
        batch = next(tr_it)
        cat, x, y = to_torch(batch)
        logits = model(cat, x)
        loss = bce_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            with torch.no_grad():
                vb = next(va_it)
                c2, x2, y2 = to_torch(vb)
                vloss = bce_loss(model(c2, x2), y2).item()
            print(f"cfg={cfg.interaction} step={step+1} val_loss={vloss:.4f}")

    with torch.no_grad():
        vb = next(va_it)
        c2, x2, y2 = to_torch(vb)
        vloss = bce_loss(model(c2, x2), y2).item()
    return vloss


def main() -> None:
    cfg1 = SMTConfig(interaction="concat", use_calibration=True)
    cfg2 = SMTConfig(interaction="dot", use_calibration=False)

    print("Training SMT config A (concat + calibration)")
    v1 = train_one(cfg1)
    print("Training SMT config B (dot, no calibration)")
    v2 = train_one(cfg2)

    print("Final val losses:")
    print("- config A:", v1)
    print("- config B:", v2)


if __name__ == "__main__":
    main()
