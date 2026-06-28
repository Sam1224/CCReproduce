from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from annotator import run_annotation_cascade
from data import (
    build_corpus,
    build_vocab,
    encode,
    make_toy_catalog,
    make_toy_queries,
    train_valid_test_split,
)
from eval import evaluate_retrieval
from mining import mine_pairs
from model import DualEncoder
from train import train_bce, train_mnr, train_triplet


def build_training_items(pairs, queries_by_id, products_by_id, vocab):
    items = []
    for it in pairs:
        q = queries_by_id[it.qid]
        p = products_by_id[it.pid]
        q_ids = encode(q.text, vocab)
        d_ids = encode(p.title, vocab)

        # BCE uses binary label: relevant if pseudo grade >= 2
        label = 1.0 if it.pseudo_grade >= 2 else 0.0
        items.append(
            {
                "q_ids": q_ids,
                "d_ids": d_ids,
                "label": label,
                "grade": it.pseudo_grade,
                "bucket": it.bucket,
                "qid": it.qid,
                "pid": it.pid,
            }
        )
    return items


def build_mnr_pairs(annotated_pairs, queries_by_id, products_by_id, vocab):
    # choose high-confidence positives per query
    best_pos = {}
    for it in annotated_pairs:
        if it.pseudo_grade < 3:
            continue
        key = it.qid
        if key not in best_pos or it.confidence > best_pos[key].confidence:
            best_pos[key] = it

    pairs = []
    for it in best_pos.values():
        q = queries_by_id[it.qid]
        p = products_by_id[it.pid]
        pairs.append((encode(q.text, vocab), encode(p.title, vocab)))
    return pairs


def build_triplets(annotated_pairs, queries_by_id, products_by_id, vocab):
    by_q = defaultdict(list)
    for it in annotated_pairs:
        by_q[it.qid].append(it)

    triplets = []
    rng = random.Random(0)
    for qid, lst in by_q.items():
        pos = [it for it in lst if it.pseudo_grade >= 3]
        neg = [it for it in lst if it.pseudo_grade <= 1 or it.bucket == "hard_neg"]
        if not pos or not neg:
            continue
        for _ in range(2):
            p_it = rng.choice(pos)
            n_it = rng.choice(neg)
            q = queries_by_id[qid]
            p = products_by_id[p_it.pid]
            n = products_by_id[n_it.pid]
            triplets.append((encode(q.text, vocab), encode(p.title, vocab), encode(n.title, vocab)))
    return triplets


def print_metrics(name: str, m: Dict[str, float]):
    msg = ", ".join([f"{k}={v:.4f}" for k, v in m.items()])
    print(f"[{name}] {msg}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_products", type=int, default=2400)
    ap.add_argument("--n_queries", type=int, default=800)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs_bce", type=int, default=2)
    ap.add_argument("--epochs_mnr", type=int, default=2)
    ap.add_argument("--epochs_triplet", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()

    products = make_toy_catalog(args.n_products, seed=args.seed)
    queries = make_toy_queries(args.n_queries, seed=args.seed + 1)

    q_train, q_valid, q_test = train_valid_test_split(queries, seed=args.seed)

    # Mining is done on train queries.
    mined, bucket_stats = mine_pairs(q_train, products, topk=args.topk, seed=args.seed)
    print("[mining] bucket sizes:", bucket_stats)

    annotated = run_annotation_cascade(mined, seed=args.seed)

    queries_by_id = {q.qid: q for q in queries}
    products_by_id = {p.pid: p for p in products}

    q_texts, p_texts = build_corpus(queries, products)
    vocab = build_vocab(list(q_texts) + list(p_texts))

    # Evaluate before training
    model = DualEncoder(vocab_size=len(vocab), dim=args.dim)
    base = evaluate_retrieval(model, vocab, q_test, products, device=args.device, k=10)
    print_metrics("baseline", base)

    train_items = build_training_items(annotated, queries_by_id, products_by_id, vocab)

    # curriculum stage 1: BCE
    train_bce(model, train_items, device=args.device, epochs=args.epochs_bce)
    after_bce = evaluate_retrieval(model, vocab, q_test, products, device=args.device, k=10)
    print_metrics("after_bce", after_bce)

    # curriculum stage 2: in-batch MNR
    mnr_pairs = build_mnr_pairs(annotated, queries_by_id, products_by_id, vocab)
    train_mnr(model, mnr_pairs, device=args.device, epochs=args.epochs_mnr)
    after_mnr = evaluate_retrieval(model, vocab, q_test, products, device=args.device, k=10)
    print_metrics("after_mnr", after_mnr)

    # curriculum stage 3: triplet
    triplets = build_triplets(annotated, queries_by_id, products_by_id, vocab)
    train_triplet(model, triplets, device=args.device, epochs=args.epochs_triplet)
    after_triplet = evaluate_retrieval(model, vocab, q_test, products, device=args.device, k=10)
    print_metrics("after_triplet", after_triplet)

    out_dir = Path(__file__).resolve().parent
    ckpt = out_dir / "ckpt.pt"
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "args": vars(args)}, ckpt)
    print(f"[saved] {ckpt}")


if __name__ == "__main__":
    main()
