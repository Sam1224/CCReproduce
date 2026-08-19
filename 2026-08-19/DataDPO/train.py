from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from data import DataBundle, accuracy, build_data, cosine_sim, softmax_loss
from model import RewardModel, TargetModel


def one_step_probe_utilities(
    target: TargetModel,
    data: DataBundle,
    *,
    lr: float = 0.4,
    max_samples: int | None = None,
    device: torch.device,
) -> torch.Tensor:
    """Estimate per-sample utility via one-step probing.

    Utility is defined as validation-loss improvement after one SGD step on a
    single sample. This is a toy proxy for the paper's target-side probing.
    """

    target.eval()

    pool_x = data.pool_x.to(device)
    pool_y = data.pool_y.to(device)
    val_x = data.val_x.to(device)
    val_y = data.val_y.to(device)

    with torch.no_grad():
        base_val_loss = softmax_loss(target(val_x), val_y).item()

    n = pool_x.shape[0]
    if max_samples is not None:
        n = min(n, max_samples)

    utilities = torch.zeros(n, dtype=torch.float32)

    for i in range(n):
        x_i = pool_x[i : i + 1]
        y_i = pool_y[i : i + 1]

        target.zero_grad(set_to_none=True)
        logits = target(x_i)
        loss = F.cross_entropy(logits, y_i)
        loss.backward()

        with torch.no_grad():
            w = target.linear.weight  # [2, d]
            b = target.linear.bias  # [2]
            gw = target.linear.weight.grad
            gb = target.linear.bias.grad

            w_new = w - lr * gw
            b_new = b - lr * gb

            val_logits = val_x @ w_new.t() + b_new
            new_val_loss = F.cross_entropy(val_logits, val_y).item()

        utilities[i] = base_val_loss - new_val_loss

    return utilities


def build_preference_pairs(
    utilities: torch.Tensor,
    *,
    num_pairs: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    n = utilities.shape[0]
    i = torch.randint(0, n, (num_pairs,), generator=g)
    j = torch.randint(0, n, (num_pairs,), generator=g)
    better = (utilities[i] > utilities[j]).float()
    return i, j, better


def train_reward_model(
    reward: RewardModel,
    data: DataBundle,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    better: torch.Tensor,
    *,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 256,
) -> None:
    reward.train()
    opt = torch.optim.Adam(reward.parameters(), lr=3e-3)

    x = data.pool_x.to(device)
    pair_i = pair_i.to(device)
    pair_j = pair_j.to(device)
    better = better.to(device)

    for _ in range(epochs):
        perm = torch.randperm(pair_i.shape[0], device=device)
        for start in range(0, pair_i.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            a = x[pair_i[idx]]
            b = x[pair_j[idx]]
            lab = better[idx]

            r_a = reward(a)
            r_b = reward(b)

            # DPO-style pairwise objective: maximize r(preferred) - r(other)
            diff = r_a - r_b
            loss = -torch.log(torch.sigmoid(diff) + 1e-8) * lab - torch.log(
                torch.sigmoid(-diff) + 1e-8
            ) * (1.0 - lab)
            loss = loss.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward.parameters(), 1.0)
            opt.step()


def greedy_select(
    *,
    features: torch.Tensor,
    base_score: torch.Tensor,
    budget: int,
    div_lambda: float = 0.25,
) -> List[int]:
    """Greedy selection with a simple diversity penalty."""

    n = features.shape[0]
    selected: List[int] = []
    chosen = torch.zeros(n, dtype=torch.bool)

    # Track max similarity to selected set for each candidate.
    max_sim = torch.zeros(n)

    for _ in range(budget):
        penalty = div_lambda * max_sim
        score = base_score - penalty
        score = score.masked_fill(chosen, -1e9)

        idx = int(torch.argmax(score).item())
        selected.append(idx)
        chosen[idx] = True

        # Update max_sim using cosine similarity to newly selected feature.
        sim = cosine_sim(features, features[idx : idx + 1]).squeeze(-1).clamp(-1, 1)
        max_sim = torch.maximum(max_sim, sim)

    return selected


def train_target_model(
    *,
    data: DataBundle,
    train_idx: torch.Tensor,
    d: int,
    seed: int,
    device: torch.device,
    epochs: int = 30,
    lr: float = 0.25,
) -> Dict[str, float]:
    g = torch.Generator().manual_seed(seed)
    model = TargetModel(d).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    x = data.pool_x[train_idx].to(device)
    y = data.pool_y[train_idx].to(device)
    val_x = data.val_x.to(device)
    val_y = data.val_y.to(device)
    test_x = data.test_x.to(device)
    test_y = data.test_y.to(device)

    for _ in range(epochs):
        perm = torch.randperm(x.shape[0], generator=g, device=device)
        for start in range(0, x.shape[0], 128):
            b = perm[start : start + 128]
            loss = F.cross_entropy(model(x[b]), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(val_x)
        test_logits = model(test_x)

    return {
        "val_acc": accuracy(val_logits, val_y),
        "test_acc": accuracy(test_logits, test_y),
        "val_loss": float(softmax_loss(val_logits, val_y).item()),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 13
    torch.manual_seed(seed)

    data = build_data(seed=seed)
    d = data.pool_x.shape[1]

    # Target model for probing.
    target = TargetModel(d).to(device)

    utilities = one_step_probe_utilities(target, data, lr=0.4, device=device)
    pair_i, pair_j, better = build_preference_pairs(utilities, num_pairs=8000, seed=seed + 3)

    reward = RewardModel(d).to(device)
    train_reward_model(reward, data, pair_i, pair_j, better, device=device)

    reward.eval()
    with torch.no_grad():
        r = reward(data.pool_x.to(device)).cpu()

    # Combine reward with external quality and a diversity term.
    # In practice, data selection also relies on external cleaning scores; here we
    # use pool_quality as a toy "clean/in-domain" signal and penalize OOD samples.
    base_score = r + 1.0 * data.pool_quality - 1.2 * (1.0 - data.pool_quality)

    budget = int(0.2 * data.pool_x.shape[0])
    per_class = budget // 2

    idx0 = torch.where(data.pool_y == 0)[0]
    idx1 = torch.where(data.pool_y == 1)[0]

    sel0_local = greedy_select(
        features=data.pool_x[idx0],
        base_score=base_score[idx0],
        budget=min(per_class, int(idx0.numel())),
        div_lambda=0.25,
    )
    sel1_local = greedy_select(
        features=data.pool_x[idx1],
        base_score=base_score[idx1],
        budget=min(per_class, int(idx1.numel())),
        div_lambda=0.25,
    )

    selected = (
        idx0[torch.tensor(sel0_local, dtype=torch.long)].tolist()
        + idx1[torch.tensor(sel1_local, dtype=torch.long)].tolist()
    )

    # If budget is odd or a class is smaller, top up by best remaining scores.
    if len(selected) < budget:
        chosen = torch.zeros(data.pool_x.shape[0], dtype=torch.bool)
        chosen[torch.tensor(selected, dtype=torch.long)] = True
        topup_score = base_score.masked_fill(chosen, -1e9)
        extra = int(torch.argmax(topup_score).item())
        selected.append(extra)

    # Stable shuffle for training.
    g_sel = torch.Generator().manual_seed(seed + 5)
    selected = torch.tensor(selected, dtype=torch.long)[
        torch.randperm(len(selected), generator=g_sel)
    ].tolist()

    g = torch.Generator().manual_seed(seed + 9)
    rand_idx = torch.randperm(data.pool_x.shape[0], generator=g)[:budget].tolist()

    metrics_selected = train_target_model(
        data=data,
        train_idx=torch.tensor(selected, dtype=torch.long),
        d=d,
        seed=seed + 100,
        device=device,
    )
    metrics_random = train_target_model(
        data=data,
        train_idx=torch.tensor(rand_idx, dtype=torch.long),
        d=d,
        seed=seed + 200,
        device=device,
    )
    metrics_full = train_target_model(
        data=data,
        train_idx=torch.arange(data.pool_x.shape[0], dtype=torch.long),
        d=d,
        seed=seed + 300,
        device=device,
    )

    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(exist_ok=True)

    torch.save({"reward": reward.state_dict()}, out_dir / "reward.pt")
    (out_dir / "selected_idx.json").write_text(
        json.dumps({"budget": budget, "idx": selected}, indent=2), encoding="utf-8"
    )

    result = {
        "budget": budget,
        "pool_size": int(data.pool_x.shape[0]),
        "selected": metrics_selected,
        "random": metrics_random,
        "full": metrics_full,
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
