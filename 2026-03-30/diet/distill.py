from __future__ import annotations

from typing import Dict, Tuple

import torch

from model import Params, batch_loss


def sgd_step(params: Params, grads: Params, lr: float) -> Params:
    """Pure functional SGD step (no in-place), keeps graph for meta-gradients."""
    return {k: params[k] - lr * grads[k] for k in params.keys()}


def meta_update_distilled_weights(
    params: Params,
    full_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    distilled_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    distilled_weights: torch.Tensor,
    inner_lr: float = 0.2,
) -> torch.Tensor:
    """One bilevel step updating distilled sample weights.

    This implements a small but faithful mechanism:

    - inner: take one gradient step on a weighted distilled batch
    - outer: evaluate loss on a full batch
    - update distilled_weights to reduce outer loss

    Note: The paper distills the dataset itself; this toy version distills **sample weights**
    for a small subset, which is a common practical approximation.
    """

    du, di, dy = distilled_batch
    fu, fi, fy = full_batch

    # Inner: weighted loss on distilled subset
    logits_loss = torch.nn.functional.binary_cross_entropy_with_logits
    d_logits = batch_loss(params, du, di, dy)

    # More explicit weighted loss so weights are used per sample.
    d_logits_each = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros_like(dy),  # placeholder, will be replaced below
        dy,
        reduction="none",
    )
    # Compute per-sample logits and loss
    from model import predict

    per_logits = predict(params, du, di)
    d_logits_each = logits_loss(per_logits, dy, reduction="none")

    w = torch.sigmoid(distilled_weights).detach() + 1e-6
    w = w / w.mean()
    inner_loss = (d_logits_each * w).mean()

    grads = torch.autograd.grad(inner_loss, params.values(), create_graph=True)
    grads_dict = {k: g for k, g in zip(params.keys(), grads)}
    params_after = sgd_step(params, grads_dict, lr=inner_lr)

    # Outer: loss on a full batch
    outer = batch_loss(params_after, fu, fi, fy)

    # Meta-grad w.r.t distilled weights
    (gw,) = torch.autograd.grad(outer, (distilled_weights,), retain_graph=False, create_graph=False)
    return gw
