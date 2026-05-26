from __future__ import annotations

import argparse

import numpy as np
import torch

from train import eval_recall_ndcg
from model import LightGCN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt["args"]
    cfg = ckpt["cfg"]

    s_tilde = torch.from_numpy(ckpt["s_tilde"]).to(args.device)

    model = LightGCN(num_nodes=cfg.num_users + cfg.num_items, dim=int(train_args["dim"]), s_tilde=s_tilde).to(args.device)
    model.set_type_masks(num_users=cfg.num_users)
    model.load_state_dict(ckpt["model"])

    metrics = eval_recall_ndcg(
        model=model,
        num_users=cfg.num_users,
        num_items=cfg.num_items,
        train_pos_by_user=ckpt["train_pos_by_user"],
        gt_by_user=ckpt["test_pos_by_user"],
        k=20,
    )

    print(metrics)


if __name__ == "__main__":
    main()
