import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from data import ToyRecConfig, build_retrieval_aggregator, generate_toy_data, set_seed
from model import ContentProjector, DiffColdDDPM, DiffusionConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="runs/diffcold")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    cfg = ToyRecConfig()
    data = generate_toy_data(cfg, seed=args.seed)

    content = torch.tensor(data["content"], dtype=torch.float32)
    item_emb_true = torch.tensor(data["item_emb_true"], dtype=torch.float32)

    warm_ids = data["warm_ids"]

    start_np = build_retrieval_aggregator(
        data["content"], warm_ids=warm_ids, warm_item_emb=data["item_emb_true"], k=cfg.k_retrieve
    )
    start = torch.tensor(start_np, dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    content = content.to(device)
    item_emb_true = item_emb_true.to(device)
    start = start.to(device)

    # baseline: simple content projection trained on warm items (regression)
    proj = ContentProjector(cfg.content_dim, cfg.embed_dim).to(device)
    proj_opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=1e-2)

    warm_content = content[warm_ids]
    warm_target = item_emb_true[warm_ids]

    for ep in range(1, 6):
        perm = torch.randperm(warm_content.size(0), device=device)
        total = 0.0
        for i in range(0, perm.numel(), args.batch_size):
            idx = perm[i : i + args.batch_size]
            pred = proj(warm_content[idx])
            loss = F.mse_loss(pred, warm_target[idx])
            proj_opt.zero_grad(set_to_none=True)
            loss.backward()
            proj_opt.step()
            total += float(loss.detach())
        print(f"[ContentProj] epoch={ep} mse={total:.4f}")

    # DiffCold DDPM
    dcfg = DiffusionConfig(embed_dim=cfg.embed_dim, content_dim=cfg.content_dim)
    ddpm = DiffColdDDPM(dcfg).to(device)
    opt = torch.optim.AdamW(ddpm.parameters(), lr=args.lr, weight_decay=1e-2)

    warm_start = start[warm_ids]

    history = []
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(warm_content.size(0), device=device)
        loss_sum = 0.0
        ddpm_sum = 0.0
        align_sum = 0.0

        ddpm.train()
        for i in range(0, perm.numel(), args.batch_size):
            idx = perm[i : i + args.batch_size]

            x0 = warm_target[idx]
            c = warm_content[idx]
            s = warm_start[idx]

            loss, parts = ddpm.loss(x0=x0, content=c, start=s)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.detach())
            ddpm_sum += parts["ddpm"]
            align_sum += parts["align"]

        row = {
            "epoch": ep,
            "loss": loss_sum,
            "ddpm": ddpm_sum,
            "align": align_sum,
        }
        history.append(row)
        print(f"[DiffCold] epoch={ep:02d} loss={loss_sum:.3f} ddpm={ddpm_sum:.3f} align={align_sum:.3f}")

    ckpt = {
        "toy_cfg": asdict(cfg),
        "diffusion_cfg": asdict(dcfg),
        "proj": proj.state_dict(),
        "ddpm": ddpm.state_dict(),
        "data": {
            "content": data["content"],
            "item_emb_true": data["item_emb_true"],
            "user_emb": data["user_emb"],
            "warm_ids": data["warm_ids"],
            "cold_ids": data["cold_ids"],
            "train_warm": data["train_warm"],
            "test_warm": data["test_warm"],
            "test_cold": data["test_cold"],
            "start": start_np,
        },
    }

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    torch.save(ckpt, ckpt_path)

    with open(os.path.join(args.out_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()
