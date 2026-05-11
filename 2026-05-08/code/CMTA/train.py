"""CMTA training -- BCE on real-vs-AIGV labels."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ToyVideoConfig, ToyVideoDataset
from model import CMTA, CMTAConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("cmta.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="outputs")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    toy = ToyVideoConfig()
    train_ds = ToyVideoDataset(toy, "train")
    cfg = CMTAConfig(feature_dim=toy.feature_dim, seq_len=toy.seq_len)
    model = CMTA(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for it, batch in enumerate(loader):
            v = batch["v"].to(args.device)
            c = batch["c"].to(args.device)
            y = batch["y"].to(args.device)
            logit = model(v, c)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
        logger.info("epoch %d avg loss %.4f", epoch, running / max(len(loader), 1))
    ckpt = Path(args.out) / "cmta.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt)
    logger.info("saved checkpoint to %s", ckpt)


if __name__ == "__main__":
    main()
