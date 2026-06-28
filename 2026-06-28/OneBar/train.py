from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data import SPECIALS, Session, build_vocab, encode
from model import QueryGenerator, pad_2d


class SessionDataset(Dataset):
    def __init__(self, samples: Sequence[Session], vocab: Dict[str, int]):
        self.samples = list(samples)
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        src = encode(s.evidence_tokens, self.vocab)
        tgt = encode(s.target_tokens, self.vocab)
        return {"src": src, "tgt": tgt, "sid": s.sid}


def collate(batch: List[dict], pad_id: int, bos_id: int, eos_id: int):
    src = pad_2d([b["src"] for b in batch], pad_id=pad_id)

    tgt_inp = []
    tgt_out = []
    for b in batch:
        t = b["tgt"]
        tgt_inp.append([bos_id] + t)
        tgt_out.append(t + [eos_id])

    tgt_inp = pad_2d(tgt_inp, pad_id=pad_id)
    tgt_out = pad_2d(tgt_out, pad_id=pad_id)

    src_pad = src.eq(pad_id)
    tgt_pad = tgt_inp.eq(pad_id)

    return {
        "src": src,
        "tgt_inp": tgt_inp,
        "tgt_out": tgt_out,
        "src_pad": src_pad,
        "tgt_pad": tgt_pad,
    }


def token_f1(pred: List[int], gold: List[int], pad_id: int, eos_id: int) -> float:
    pred = [t for t in pred if t not in (pad_id, eos_id)]
    gold = [t for t in gold if t not in (pad_id, eos_id)]
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    pset = {}
    for t in pred:
        pset[t] = pset.get(t, 0) + 1
    gset = {}
    for t in gold:
        gset[t] = gset.get(t, 0) + 1
    inter = 0
    for t, c in pset.items():
        inter += min(c, gset.get(t, 0))
    prec = inter / max(1, len(pred))
    rec = inter / max(1, len(gold))
    if prec + rec <= 1e-9:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def train_supervised(
    model: QueryGenerator,
    train_samples: Sequence[Session],
    vocab: Dict[str, int],
    device: str = "cpu",
    epochs: int = 2,
    lr: float = 2e-3,
    batch_size: int = 64,
):
    pad_id = vocab["<pad>"]
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]

    ds = SessionDataset(train_samples, vocab)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id),
    )

    model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    for _ in range(epochs):
        for batch in dl:
            src = batch["src"].to(device)
            tgt_inp = batch["tgt_inp"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            src_pad = batch["src_pad"].to(device)
            tgt_pad = batch["tgt_pad"].to(device)

            logits = model(src, tgt_inp, src_pad, tgt_pad)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def train_preference_internalization(
    model: QueryGenerator,
    train_samples: Sequence[Session],
    vocab: Dict[str, int],
    device: str = "cpu",
    epochs: int = 1,
    lr: float = 8e-4,
    batch_size: int = 64,
):
    """PIOPD-like on-policy self-distillation.

    We generate queries with the current model (on-policy), compute a reward
    proxy (token-F1 vs target), then weight the supervised loss by reward.

    This keeps the workflow training-free regarding an external reward model,
    while still nudging the generator toward behavior-consistent outputs.
    """

    pad_id = vocab["<pad>"]
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]

    ds = SessionDataset(train_samples, vocab)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id),
    )

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="none")

    for _ in range(epochs):
        model.train()
        for batch in dl:
            src = batch["src"].to(device)
            tgt_inp = batch["tgt_inp"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            src_pad = batch["src_pad"].to(device)
            tgt_pad = batch["tgt_pad"].to(device)

            with torch.no_grad():
                gen = model.greedy_decode(src, src_pad, bos_id=bos_id, eos_id=eos_id)
                # compute reward per sample
                rewards = []
                for i in range(gen.size(0)):
                    rewards.append(
                        token_f1(
                            pred=gen[i].tolist(),
                            gold=tgt_out[i].tolist(),
                            pad_id=pad_id,
                            eos_id=eos_id,
                        )
                    )
                w = torch.tensor(rewards, dtype=torch.float32, device=device).clamp(0.0, 1.0)

            logits = model(src, tgt_inp, src_pad, tgt_pad)
            per_tok = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)).view(tgt_out.size(0), -1)
            per_ex = per_tok.mean(dim=1)
            loss = (per_ex * w).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
