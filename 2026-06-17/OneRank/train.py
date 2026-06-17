from __future__ import annotations

"""Train the toy OneRank model with a multi-task list-wise InfoNCE + point-wise
BCE objective. Prints a decreasing loss and saves a checkpoint."""

import argparse
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from data import ToyConfig, generate_dataset, iterate_batches
from model import OneRank, OneRankConfig, N_TASKS


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensors(batch, device):
    return (
        torch.as_tensor(batch["hist"], dtype=torch.long, device=device),
        torch.as_tensor(batch["anchor"], dtype=torch.long, device=device),
        torch.as_tensor(batch["cands"], dtype=torch.long, device=device),
        torch.as_tensor(batch["labels"], dtype=torch.float32, device=device),  # [B,N,3]
    )


def multitask_loss(scores, labels, infonce_w: float = 1.0):
    """scores: [B, 3, N] logits ; labels: [B, N, 3] (click, cart, order).

    point-wise BCE over all (task, candidate) pairs +
    list-wise InfoNCE: per task, softmax over candidates against the
    relevance-normalized positive distribution (masked when no positive).
    """
    lab = labels.permute(0, 2, 1).contiguous()                   # [B, 3, N]

    # point-wise BCE
    bce = F.binary_cross_entropy_with_logits(scores, lab)

    # list-wise InfoNCE per task
    log_prob = F.log_softmax(scores, dim=-1)                      # [B, 3, N]
    pos_count = lab.sum(dim=-1)                                   # [B, 3]
    target = lab / pos_count.clamp(min=1.0).unsqueeze(-1)        # normalized target dist
    nce_per = -(target * log_prob).sum(dim=-1)                   # [B, 3]
    valid = (pos_count > 0).float()
    infonce = (nce_per * valid).sum() / valid.sum().clamp(min=1.0)

    return bce + infonce_w * infonce, bce.item(), infonce.item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="runs/onerank")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--infonce_w", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    toy = ToyConfig()
    data = generate_dataset(toy, seed=args.seed)
    train = data["train"]

    mcfg = OneRankConfig(
        n_items=toy.n_items, hist_len=toy.hist_len, anchor_len=toy.anchor_len,
        n_candidates=toy.n_candidates,
    )
    model = OneRank(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"device={device}  params={sum(p.numel() for p in model.parameters()):,}")
    model.train()
    for ep in range(1, args.epochs + 1):
        tot, tot_bce, tot_nce, nb = 0.0, 0.0, 0.0, 0
        for batch in iterate_batches(train, args.batch_size, seed=args.seed + ep):
            hist, anchor, cands, labels = to_tensors(batch, device)
            scores = model(hist, anchor, cands)
            loss, bce, nce = multitask_loss(scores, labels, args.infonce_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); tot_bce += bce; tot_nce += nce; nb += 1
        print(f"epoch={ep:02d}  loss={tot/nb:.4f}  (bce={tot_bce/nb:.4f}  infonce={tot_nce/nb:.4f})")

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    torch.save({
        "model_state": model.state_dict(),
        "model_cfg": asdict(mcfg),
        "toy_cfg": asdict(toy),
        "seed": args.seed,
    }, ckpt_path)
    print(f"\nsaved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
