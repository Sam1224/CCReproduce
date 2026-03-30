from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from dataset import make_toy_implicit_data, split_train_val, stream_batches
from distill import meta_update_distilled_weights
from model import MFConfig, batch_loss, init_params


def to_torch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(batch.user).long(),
        torch.from_numpy(batch.item).long(),
        torch.from_numpy(batch.label).float(),
    )


def main() -> None:
    n_users = 200
    n_items = 300
    data = make_toy_implicit_data(n_users=n_users, n_items=n_items, n_pos=5000, seed=0)
    train, val = split_train_val(data, val_ratio=0.2, seed=0)

    cfg = MFConfig(n_users=n_users, n_items=n_items, dim=32)
    params = init_params(cfg, seed=0)

    # Choose a small distilled subset from the training data.
    rng = np.random.default_rng(0)
    distilled_idx = rng.choice(len(train), size=256, replace=False)
    distilled = [train[i] for i in distilled_idx]

    # Learnable per-sample weights for the distilled subset.
    distilled_weights = torch.zeros(len(distilled), requires_grad=True)

    outer_stream = stream_batches(train, n_users, n_items, batch_size=256, neg_ratio=1, seed=1)
    d_stream = stream_batches(distilled, n_users, n_items, batch_size=128, neg_ratio=1, seed=2)

    meta_lr = 0.1
    train_lr = 0.5

    for step in range(200):
        full_b = to_torch(next(outer_stream))
        d_b = to_torch(next(d_stream))

        # Meta-update weights
        gw = meta_update_distilled_weights(
            params=params,
            full_batch=full_b,
            distilled_batch=d_b,
            distilled_weights=distilled_weights,
            inner_lr=0.2,
        )
        with torch.no_grad():
            distilled_weights -= meta_lr * gw

        # Train model params on distilled batch (weighted)
        du, di, dy = d_b
        from model import predict

        logits = predict(params, du, di)
        per_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, dy, reduction="none")
        w = torch.sigmoid(distilled_weights[: per_loss.shape[0]]).detach() + 1e-6
        w = w / w.mean()
        loss = (per_loss * w).mean()

        grads = torch.autograd.grad(loss, params.values())
        with torch.no_grad():
            for (k, p), g in zip(list(params.items()), grads):
                params[k] = (p - train_lr * g).detach().requires_grad_(True)

        if (step + 1) % 50 == 0:
            # quick validation loss
            vb = to_torch(next(stream_batches(val, n_users, n_items, batch_size=256, neg_ratio=1, seed=10 + step)))
            vloss = batch_loss(params, vb[0], vb[1], vb[2]).item()
            print(f"step={step+1} val_loss={vloss:.4f}")


if __name__ == "__main__":
    main()
