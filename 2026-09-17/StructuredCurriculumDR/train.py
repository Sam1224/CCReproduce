from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data import (
    DataConfig,
    build_click_like_pairs,
    build_corpus,
    build_mnr_pairs,
    build_training_items,
    build_triplets,
    build_vocab,
    build_cfg_from_ckpt,
    evaluate_retrieval,
    make_toy_catalog,
    make_toy_queries,
    mine_pairs,
    run_annotation_cascade,
    train_valid_test_split,
)
from model import Batch, DualEncoder, collate_pair_batch, make_mask, pad_2d

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
BASELINE_CKPT = CKPT_DIR / "baseline.pt"
CURRICULUM_CKPT = CKPT_DIR / "structured_curriculum.pt"


class PairDataset(Dataset):
    def __init__(self, items: Sequence[Dict]):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def train_bce(model: DualEncoder, train_items: Sequence[Dict], device: str = "cpu", lr: float = 2e-3, epochs: int = 2, batch_size: int = 128):
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


def train_mnr(model: DualEncoder, pairs: Sequence[Tuple[List[int], List[int]]], device: str = "cpu", lr: float = 1e-3, epochs: int = 2, batch_size: int = 64):
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    items = list(pairs)
    for _ in range(epochs):
        import random

        random.Random(0).shuffle(items)
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
            logits = q @ d.T
            target = torch.arange(len(batch), device=device)
            loss = nn.CrossEntropyLoss()(logits, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def train_triplet(model: DualEncoder, triplets: Sequence[Tuple[List[int], List[int], List[int]]], device: str = "cpu", lr: float = 8e-4, epochs: int = 1, batch_size: int = 64, margin: float = 0.2):
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    items = list(triplets)
    for _ in range(epochs):
        import random

        random.Random(0).shuffle(items)
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


def save_checkpoint(path: Path, model: DualEncoder, cfg: DataConfig, vocab: Dict[str, int], dim: int) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"data_cfg": cfg.__dict__, "vocab": vocab, "dim": dim, "state_dict": model.state_dict()}, path)
    print(f"saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--epochs-bce", type=int, default=2)
    ap.add_argument("--epochs-mnr", type=int, default=2)
    ap.add_argument("--epochs-triplet", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = DataConfig(seed=args.seed)
    products = make_toy_catalog(cfg.n_products, seed=cfg.seed)
    queries = make_toy_queries(cfg.n_queries, seed=cfg.seed + 1)
    q_train, q_valid, q_test = train_valid_test_split(queries, seed=cfg.seed)
    q_texts, p_texts = build_corpus(queries, products)
    vocab = build_vocab(list(q_texts) + list(p_texts))

    baseline_items = build_click_like_pairs(q_train, products, vocab, seed=cfg.seed)
    baseline = DualEncoder(vocab_size=len(vocab), dim=args.dim)
    train_bce(baseline, baseline_items, device=args.device, epochs=max(2, args.epochs_bce + 1))
    baseline_metrics = evaluate_retrieval(baseline, vocab, q_test, products, device=args.device, k=10)
    print("baseline:", {k: round(v, 4) for k, v in baseline_metrics.items()})
    save_checkpoint(BASELINE_CKPT, baseline, cfg, vocab, args.dim)

    mined, bucket_stats = mine_pairs(q_train, products, topk=cfg.topk, seed=cfg.seed)
    print("mined bucket sizes:", bucket_stats)
    annotated = run_annotation_cascade(mined, seed=cfg.seed)
    queries_by_id = {q.qid: q for q in queries}
    products_by_id = {p.pid: p for p in products}
    curriculum_items = build_training_items(annotated, queries_by_id, products_by_id, vocab)
    mnr_pairs = build_mnr_pairs(annotated, queries_by_id, products_by_id, vocab)
    triplets = build_triplets(annotated, queries_by_id, products_by_id, vocab)

    model = DualEncoder(vocab_size=len(vocab), dim=args.dim)
    train_bce(model, curriculum_items, device=args.device, epochs=args.epochs_bce)
    print("after_bce:", {k: round(v, 4) for k, v in evaluate_retrieval(model, vocab, q_test, products, device=args.device, k=10).items()})
    train_mnr(model, mnr_pairs, device=args.device, epochs=args.epochs_mnr)
    print("after_mnr:", {k: round(v, 4) for k, v in evaluate_retrieval(model, vocab, q_test, products, device=args.device, k=10).items()})
    train_triplet(model, triplets, device=args.device, epochs=args.epochs_triplet)
    final_metrics = evaluate_retrieval(model, vocab, q_test, products, device=args.device, k=10)
    print("after_triplet:", {k: round(v, 4) for k, v in final_metrics.items()})
    save_checkpoint(CURRICULUM_CKPT, model, cfg, vocab, args.dim)


if __name__ == "__main__":
    main()
