from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader

from rodpo.data import NextItemDataset, ToySeqData, collate_next_item, split_sequences
from rodpo.model import ModelConfig, SeqRecModel


def ndcg_at_k(rank: int, k: int) -> float:
    if rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def mrr_at_k(rank: int, k: int) -> float:
    if rank > k:
        return 0.0
    return 1.0 / rank


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    data = ToySeqData.load(args.data)
    train_seqs, val_seqs, test_seqs = split_sequences(data.sequences)
    test_ds = NextItemDataset(test_seqs, max_seq_len=args.max_seq_len)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_next_item)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig(num_items=data.num_items, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    model = SeqRecModel(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    k = args.k
    total = 0
    ndcg = 0.0
    mrr = 0.0

    for seq, pos in test_dl:
        seq = seq.to(device)
        pos = pos.to(device)
        scores = model.score_all_items(seq)
        order = torch.argsort(scores, dim=-1, descending=True)
        ranks = (order == pos.unsqueeze(-1)).nonzero(as_tuple=False)[:, 1] + 1

        for r in ranks.tolist():
            total += 1
            ndcg += ndcg_at_k(r, k)
            mrr += mrr_at_k(r, k)

    print({"NDCG@k": ndcg / max(1, total), "MRR@k": mrr / max(1, total), "k": k, "n": total})


if __name__ == "__main__":
    main()
