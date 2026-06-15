from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from data import ToyConfig, build_bce_samples, build_test_targets, generate_dataset, set_seed
from denoise import build_popularity_gate, pad_weight
from metrics import coverage_at_k, eval_ranking, gini_diversity
from model import MatrixFactorization, ModelConfig


def train_one(
    name: str,
    cfg: ToyConfig,
    data: dict,
    alpha: float,
    eta: float,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> dict:
    u, i, y = build_bce_samples(cfg, data, seed=seed)

    item_pop = data["item_pop"]
    gate_s = build_popularity_gate(item_pop, eta=eta)

    mcfg = ModelConfig(n_users=cfg.n_users, n_items=cfg.n_items, dim=cfg.latent_dim)
    model = MatrixFactorization(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    u_t = torch.tensor(u, dtype=torch.long, device=device)
    i_t = torch.tensor(i, dtype=torch.long, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    model.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(u_t.size(0), device=device)
        total = 0.0
        for s in range(0, perm.numel(), batch_size):
            idx = perm[s : s + batch_size]
            pred = model(u_t[idx], i_t[idx])

            # compute weights on cpu numpy to keep it explicit and readable
            w = pad_weight(
                y_hat=pred.detach().cpu().numpy(),
                y_true=y_t[idx].detach().cpu().numpy(),
                gate_s=gate_s,
                alpha=alpha,
                item_ids=i_t[idx].detach().cpu().numpy(),
            )
            w_t = torch.tensor(w, dtype=torch.float32, device=device)

            loss = F.binary_cross_entropy(pred, y_t[idx], weight=w_t, reduction="mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())

        print(f"[{name}] epoch={ep:02d} loss={total:.3f}")

    # eval: score all items for each user
    model.eval()
    with torch.no_grad():
        user_ids = torch.arange(cfg.n_users, device=device, dtype=torch.long)
        item_ids = torch.arange(cfg.n_items, device=device, dtype=torch.long)
        u_mat = user_ids.unsqueeze(1).expand(-1, cfg.n_items).reshape(-1)
        i_mat = item_ids.unsqueeze(0).expand(cfg.n_users, -1).reshape(-1)
        scores = model(u_mat, i_mat).reshape(cfg.n_users, cfg.n_items).cpu().numpy()

    targets = build_test_targets(cfg, data)
    rank = eval_ranking(scores, targets, ks=(20, 50))

    k = 50
    topk = np.argsort(-scores, axis=1)[:, :k]
    coverage = coverage_at_k(topk, cfg.n_items)
    gini = gini_diversity(topk, cfg.n_items)

    return {
        "name": name,
        "alpha": alpha,
        "eta": eta,
        **rank,
        "coverage@50": coverage,
        "gini_div": gini,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="runs/pad")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-3)

    # PAD hypers
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--eta", type=float, default=2.0)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    cfg = ToyConfig(latent_dim=32)
    data = generate_dataset(cfg, seed=args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ERM == alpha=0, eta=0
    m1 = train_one(
        "ERM",
        cfg,
        data,
        alpha=0.0,
        eta=0.0,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # Base denoise (RCE-style): alpha>0, eta=0
    m2 = train_one(
        "BaseDenoise(RCE)",
        cfg,
        data,
        alpha=args.alpha,
        eta=0.0,
        seed=args.seed + 1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # PAD: alpha>0, eta>0
    m3 = train_one(
        "PAD",
        cfg,
        data,
        alpha=args.alpha,
        eta=args.eta,
        seed=args.seed + 2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    metrics = {"ERM": m1, "BaseDenoise": m2, "PAD": m3}

    print("\n=== Summary (toy) ===")
    for k, v in metrics.items():
        print(
            f"{k}: recall@50={v['recall@50']:.4f} ndcg@50={v['ndcg@50']:.4f} "
            f"coverage@50={v['coverage@50']:.4f} gini_div={v['gini_div']:.4f} (alpha={v['alpha']}, eta={v['eta']})"
        )

    ckpt = {
        "toy_cfg": asdict(cfg),
        "metrics": metrics,
    }

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    torch.save(ckpt, ckpt_path)

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nsaved: {ckpt_path}")


if __name__ == "__main__":
    main()
