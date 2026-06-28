from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import Batch, DualEncoder, collate_pair_batch


class PairDataset(Dataset):
    def __init__(self, items: Sequence[Dict]):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def train_bce(
    model: DualEncoder,
    train_items: Sequence[Dict],
    device: str = "cpu",
    lr: float = 2e-3,
    epochs: int = 2,
    batch_size: int = 128,
) -> None:
    model.to(device)
    model.train()

    ds = PairDataset(train_items)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pair_batch)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        for batch in dl:
            q_ids = batch["q_ids"].to(device)
            d_ids = batch["d_ids"].to(device)
            q_mask = batch["q_mask"].to(device)
            d_mask = batch["d_mask"].to(device)
            y = batch["label"].to(device)

            sims = model(Batch(q_ids=q_ids, d_ids=d_ids, q_mask=q_mask, d_mask=d_mask))
            loss = loss_fn(sims, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def train_mnr(
    model: DualEncoder,
    pairs: Sequence[Tuple[List[int], List[int]]],
    device: str = "cpu",
    lr: float = 1e-3,
    epochs: int = 2,
    batch_size: int = 64,
):
    """Multi-Negative Ranking / In-batch InfoNCE.

    pairs: list of (q_ids, pos_d_ids)
    """

    from model import make_mask, pad_2d

    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    rng = random.Random(0)
    items = list(pairs)

    for _ in range(epochs):
        rng.shuffle(items)
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            if len(batch) < 2:
                continue
            q_pad = pad_2d([b[0] for b in batch], pad_id=0).to(device)
            d_pad = pad_2d([b[1] for b in batch], pad_id=0).to(device)
            q_mask = make_mask(q_pad).to(device)
            d_mask = make_mask(d_pad).to(device)

            q = model.encode_query(q_pad, q_mask)
            d = model.encode_doc(d_pad, d_mask)
            logits = q @ d.T  # [B, B]
            target = torch.arange(len(batch), device=device)
            loss = nn.CrossEntropyLoss()(logits, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def train_triplet(
    model: DualEncoder,
    triplets: Sequence[Tuple[List[int], List[int], List[int]]],
    device: str = "cpu",
    lr: float = 8e-4,
    epochs: int = 1,
    batch_size: int = 64,
    margin: float = 0.2,
):
    from model import make_mask, pad_2d

    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    rng = random.Random(0)
    items = list(triplets)

    for _ in range(epochs):
        rng.shuffle(items)
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            if not batch:
                continue
            q_pad = pad_2d([b[0] for b in batch], pad_id=0).to(device)
            p_pad = pad_2d([b[1] for b in batch], pad_id=0).to(device)
            n_pad = pad_2d([b[2] for b in batch], pad_id=0).to(device)

            q_mask = make_mask(q_pad).to(device)
            p_mask = make_mask(p_pad).to(device)
            n_mask = make_mask(n_pad).to(device)

            q = model.encode_query(q_pad, q_mask)
            p = model.encode_doc(p_pad, p_mask)
            n = model.encode_doc(n_pad, n_mask)

            pos = (q * p).sum(dim=-1)
            neg = (q * n).sum(dim=-1)
            loss = torch.relu(margin - pos + neg).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    return ap.parse_args()


if __name__ == "__main__":
    # training is orchestrated by run_pipeline.py
    parse_args()
