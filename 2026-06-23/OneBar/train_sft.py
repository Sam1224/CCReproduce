"""
train_sft.py — Stage 1: Content-Grounded SFT (paper Sec. 3.4.1).

Cross-entropy on clicked queries y+ (and [REJECT] for non-compliant triggers).
Runs on CPU with a tiny config for smoke testing.

Example (smoke test):
    python train_sft.py --tiny --max_steps 3 --batch_size 2 \
        --save_dir outputs/sft
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from data import generate_dataset, SFTDataset
from model import OneBarGenerator
from losses import sft_loss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="facebook/bart-base")
    ap.add_argument("--tiny", action="store_true",
                    help="use a tiny random BART (fast CPU smoke test)")
    ap.add_argument("--offline", action="store_true",
                    help="never hit the HF hub; fall back to tiny random BART")
    ap.add_argument("--n_train", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--prompt_style", default="compressed",
                    choices=["compressed", "verbose"])
    ap.add_argument("--save_dir", default="outputs/sft")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model = OneBarGenerator(args.model_name, tiny=args.tiny, offline=args.offline)
    tok = model.tokenizer

    samples = generate_dataset(n=args.n_train, seed=args.seed)
    ds = SFTDataset(samples, tok, prompt_style=args.prompt_style)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            # HF returns CE loss directly, but we recompute with losses.sft_loss
            # to make the objective explicit and reusable (Eq. 5).
            loss = sft_loss(out.logits, batch["labels"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            print(f"[SFT] epoch {epoch} step {step} loss {loss.item():.4f}")
            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break

    os.makedirs(args.save_dir, exist_ok=True)
    model.model.save_pretrained(args.save_dir)
    tok.save_pretrained(args.save_dir)
    print(f"[SFT] saved checkpoint -> {args.save_dir}")


if __name__ == "__main__":
    main()
