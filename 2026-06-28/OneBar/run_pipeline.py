from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import build_vocab, make_sessions, split
from eval import evaluate
from model import QueryGenerator
from train import train_preference_internalization, train_supervised


def print_metrics(name, m):
    msg = ", ".join([f"{k}={v:.4f}" for k, v in m.items()])
    print(f"[{name}] {msg}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs_sup", type=int, default=2)
    ap.add_argument("--epochs_pi", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()

    samples = make_sessions(args.n, seed=args.seed)
    tr, va, te = split(samples, seed=args.seed)

    vocab = build_vocab(samples)
    model = QueryGenerator(vocab_size=len(vocab), d_model=128, nhead=4, num_layers=2)
    model.to(args.device)

    # before training
    print_metrics("baseline", evaluate(model, te[:400], vocab, device=args.device))

    # supervised
    train_supervised(model, tr, vocab, device=args.device, epochs=args.epochs_sup)
    print_metrics("after_sup", evaluate(model, te[:400], vocab, device=args.device))

    # preference internalization
    train_preference_internalization(model, tr, vocab, device=args.device, epochs=args.epochs_pi)
    print_metrics("after_pi", evaluate(model, te[:400], vocab, device=args.device))

    out = Path(__file__).resolve().parent / "ckpt.pt"
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "args": vars(args)}, out)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
