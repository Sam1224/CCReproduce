from __future__ import annotations

import argparse
import copy
from typing import Tuple

import numpy as np
import torch
from torch import nn

from .calibration import calibrate_tau, evaluate_policy, probe_probs
from .data import SyntheticDataConfig, make_splits
from .models import ConfidenceProbe
from .utils import default_device, set_seed


def adapt_on_instance(
    base_model: ConfidenceProbe,
    x: torch.Tensor,
    c: torch.Tensor,
    mask: torch.Tensor,
    lr: float = 5e-2,
    steps: int = 1,
    prefix_k: int = 4,
) -> ConfidenceProbe:
    """ORCA-style per-instance test-time training.

    We clone the probe and run a few gradient steps on a short prefix using
    pseudo-labels c (standing in for self-consistency labels).
    """

    model = copy.deepcopy(base_model)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    x_p = x[:, :prefix_k]
    c_p = c[:, :prefix_k]
    m_p = mask[:, :prefix_k]

    for _ in range(steps):
        logits = model(x_p)
        loss = (bce(logits, c_p) * m_p).sum() / m_p.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    return model


def orca_probs(
    base_model: ConfidenceProbe,
    dataset,
    device: torch.device,
    lr: float,
    steps: int,
    prefix_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    probs = []
    ys = []

    for item in dataset:
        x = item["x"].unsqueeze(0).to(device)
        y = item["y"].unsqueeze(0).cpu().numpy()
        c = item["c"].unsqueeze(0).to(device)
        m = item["mask"].unsqueeze(0).to(device)

        adapted = adapt_on_instance(base_model, x, c, m, lr=lr, steps=steps, prefix_k=prefix_k)
        with torch.no_grad():
            p = adapted(x).sigmoid().cpu().numpy()

        probs.append(p)
        ys.append(y)

    return np.concatenate(probs, axis=0), np.concatenate(ys, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_instances", type=int, default=8000)
    parser.add_argument("--max_steps", type=int, default=32)
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--domain_shift", type=float, default=0.6, help="simulate OOD")

    parser.add_argument("--ttt_lr", type=float, default=5e-2)
    parser.add_argument("--ttt_steps", type=int, default=1)
    parser.add_argument("--ttt_prefix_k", type=int, default=4)

    args = parser.parse_args()
    set_seed(args.seed)
    device = default_device()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = ConfidenceProbe(feature_dim=ckpt["feature_dim"], hidden_dim=ckpt["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cfg = SyntheticDataConfig(
        num_instances=args.num_instances,
        max_steps=args.max_steps,
        feature_dim=args.feature_dim,
        domain_shift=args.domain_shift,
        seed=args.seed,
    )
    _, cal_ds, test_ds = make_splits(cfg)

    # --- Static procedure ---
    p_cal, y_cal = probe_probs(model, cal_ds, device)
    tau_static, cal_stats_static = calibrate_tau(p_cal, y_cal, delta=args.delta)

    p_test, y_test = probe_probs(model, test_ds, device)
    test_stats_static = evaluate_policy(p_test, y_test, tau=tau_static)

    # --- ORCA procedure (TTT + calibration on the procedure) ---
    p_cal_orca, y_cal_orca = orca_probs(
        model,
        cal_ds,
        device=device,
        lr=args.ttt_lr,
        steps=args.ttt_steps,
        prefix_k=args.ttt_prefix_k,
    )
    tau_orca, cal_stats_orca = calibrate_tau(p_cal_orca, y_cal_orca, delta=args.delta)

    p_test_orca, y_test_orca = orca_probs(
        model,
        test_ds,
        device=device,
        lr=args.ttt_lr,
        steps=args.ttt_steps,
        prefix_k=args.ttt_prefix_k,
    )
    test_stats_orca = evaluate_policy(p_test_orca, y_test_orca, tau=tau_orca)

    print("\n=== Static calibration ===")
    print(f"tau: {tau_static:.3f}")
    print(f"cal  risk={cal_stats_static.risk:.3f} savings={cal_stats_static.savings:.3f} avg_stop={cal_stats_static.avg_stop_step:.2f}")
    print(f"test risk={test_stats_static.risk:.3f} savings={test_stats_static.savings:.3f} avg_stop={test_stats_static.avg_stop_step:.2f}")

    print("\n=== ORCA (TTT + procedure calibration) ===")
    print(f"tau: {tau_orca:.3f}")
    print(f"cal  risk={cal_stats_orca.risk:.3f} savings={cal_stats_orca.savings:.3f} avg_stop={cal_stats_orca.avg_stop_step:.2f}")
    print(f"test risk={test_stats_orca.risk:.3f} savings={test_stats_orca.savings:.3f} avg_stop={test_stats_orca.avg_stop_step:.2f}")


if __name__ == "__main__":
    main()
