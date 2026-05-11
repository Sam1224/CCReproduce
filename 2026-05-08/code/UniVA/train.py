"""UniVA training loop -- SFT (next-SID CE) + eCPM-aware RL term.

Run:
    python train.py --epochs 3 --batch-size 64
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ToyAdRecDataset, ToyConfig
from model import CommercialSIDTokenizer, GARDecoder, UniVAConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("univa.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="outputs")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def step(decoder: GARDecoder, tokenizer: CommercialSIDTokenizer, batch: dict, device: str) -> dict:
    history = batch["history"].to(device)
    target = batch["target"].to(device)
    reward = batch["reward"].to(device)
    target_codes = tokenizer.encode(target)
    logits_per_level, value = decoder(history, target_codes)
    # SFT term: per-level CE.
    sft = sum(F.cross_entropy(logits, target_codes[:, level]) for level, logits in enumerate(logits_per_level))
    # eCPM-aware RL term: regress decoder's value head onto observed reward.
    # TODO[paper]: full PPO with eCPM oracle. Eq. (RL): J_RL = E[r * log p(SID|h)].
    rl = F.mse_loss(value, reward)
    loss = sft + decoder.cfg.value_loss_weight * rl
    return {"loss": loss, "sft": sft.detach(), "rl": rl.detach()}


def main() -> None:
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    toy = ToyConfig()
    train_ds = ToyAdRecDataset(toy, "train")
    eval_ds = ToyAdRecDataset(toy, "eval")
    item_meta = train_ds.item_value_meta()

    cfg = UniVAConfig(
        num_items=toy.num_items,
        num_ecpm_buckets=toy.num_ecpm_buckets,
        num_margin_buckets=toy.num_margin_buckets,
        history_len=toy.history_len,
    )
    tokenizer = CommercialSIDTokenizer(cfg, item_meta)
    decoder = GARDecoder(cfg)
    model = torch.nn.ModuleDict({"tok": tokenizer, "dec": decoder}).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        for it, batch in enumerate(loader):
            out = step(decoder, tokenizer, batch, args.device)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            if it % 16 == 0:
                logger.info(
                    "epoch=%d it=%d loss=%.4f sft=%.4f rl=%.4f",
                    epoch, it, out["loss"].item(), out["sft"].item(), out["rl"].item(),
                )

    ckpt = Path(args.out) / "univa.pt"
    torch.save({"tokenizer": tokenizer.state_dict(), "decoder": decoder.state_dict(), "cfg": cfg.__dict__}, ckpt)
    logger.info("saved checkpoint to %s", ckpt)


if __name__ == "__main__":
    main()
