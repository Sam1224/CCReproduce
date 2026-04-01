from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rodpo.data import NextItemDataset, ToySeqData, collate_next_item, split_sequences
from rodpo.dpo import dpo_loss, sample_negative_items
from rodpo.model import ModelConfig, SeqRecModel


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stage", choices=["ce", "dpo"], default="ce")
    parser.add_argument("--init-from", default="")
    parser.add_argument("--ref-from", default="")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)

    parser.add_argument("--use-sparse-moe", action="store_true")
    parser.add_argument("--moe-num-experts", type=int, default=4)

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--neg-sampling", choices=["random", "hard", "topk"], default="topk")
    parser.add_argument("--topk", type=int, default=50)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = ToySeqData.load(args.data)
    train_seqs, val_seqs, _ = split_sequences(data.sequences)

    train_ds = NextItemDataset(train_seqs, max_seq_len=args.max_seq_len)
    val_ds = NextItemDataset(val_seqs, max_seq_len=args.max_seq_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_next_item)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_next_item)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ModelConfig(
        num_items=data.num_items,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        use_sparse_moe=args.use_sparse_moe,
        moe_num_experts=args.moe_num_experts,
    )
    model = SeqRecModel(cfg).to(device)

    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=device))

    ref_model = None
    if args.stage == "dpo":
        if not args.ref_from:
            raise ValueError("--ref-from is required for DPO stage")
        ref_model = SeqRecModel(cfg).to(device)
        ref_model.load_state_dict(torch.load(args.ref_from, map_location=device))
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"train[{args.stage}] epoch {epoch}")
        total_loss = 0.0

        for seq, pos in pbar:
            seq = seq.to(device)
            pos = pos.to(device)

            scores = model.score_all_items(seq)

            if args.stage == "ce":
                loss = F.cross_entropy(scores, pos)
            else:
                neg = sample_negative_items(scores, pos, num_items=data.num_items, strategy=args.neg_sampling, topk=args.topk)
                logp_pos = model.score_items(seq, pos)
                logp_neg = model.score_items(seq, neg)
                with torch.no_grad():
                    logp_pos_ref = ref_model.score_items(seq, pos)
                    logp_neg_ref = ref_model.score_items(seq, neg)
                loss = dpo_loss(logp_pos, logp_neg, logp_pos_ref, logp_neg_ref, beta=args.beta)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            loss_value = float(loss.detach())
            total_loss += loss_value
            pbar.set_postfix(loss=loss_value)

        avg_train = total_loss / max(1, len(train_dl))

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for seq, pos in val_dl:
                seq = seq.to(device)
                pos = pos.to(device)
                scores = model.score_all_items(seq)
                val_loss += float(F.cross_entropy(scores, pos))
            avg_val = val_loss / max(1, len(val_dl))

        history.append({"epoch": epoch, "train_loss": avg_train, "val_ce_loss": avg_val})

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), out_dir / "model.pt")

    save_json(str(out_dir / "history.json"), history)
    save_json(str(out_dir / "config.json"), vars(args))


if __name__ == "__main__":
    main()
