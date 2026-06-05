from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import CatalogItem, UserExample, default_synthetic_setup
from model import QueryAgentR1Toy, Vocab


def build_item_token_tensor(vocab: Vocab, catalog: list[CatalogItem]) -> torch.Tensor:
    ids = [vocab.encode(it.tokens) for it in catalog]
    return torch.tensor(ids, dtype=torch.long)


def build_history_token_tensor(item_token_ids: torch.Tensor, histories: list[list[int]]) -> torch.Tensor:
    tokens = [item_token_ids[torch.tensor(h, dtype=torch.long)] for h in histories]
    return torch.stack(tokens, dim=0)


def build_target_query_tensor(vocab: Vocab, targets: list[list[str]], max_len: int) -> torch.Tensor:
    tok = [vocab.encode(t)[:max_len] for t in targets]
    tok = [t + [vocab.token_to_id["<pad>"]] * (max_len - len(t)) for t in tok]
    return torch.tensor(tok, dtype=torch.long)


def load_checkpoint(path: str) -> tuple[dict, Vocab]:
    ckpt = torch.load(path, map_location="cpu")
    vocab = Vocab(token_to_id=ckpt["vocab"]["token_to_id"], id_to_token=ckpt["vocab"]["id_to_token"])
    return ckpt["model"], vocab


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/queryagent_r1.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    catalog, _, _, test = default_synthetic_setup(seed=args.seed)

    model_state, vocab = load_checkpoint(args.ckpt)
    item_token_ids = build_item_token_tensor(vocab, catalog)

    device = torch.device(args.device)
    model = QueryAgentR1Toy(
        n_items=len(catalog),
        vocab_size=len(vocab.id_to_token),
        embed_dim=args.embed_dim,
        item_token_ids=item_token_ids,
        max_query_len=2,
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()

    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=lambda x: x)

    q_em_list = []
    i_hit_list = []
    cons_list = []

    pad = vocab.token_to_id["<pad>"]

    for batch in loader:
        histories = [ex.history for ex in batch]
        targets = [ex.target_query_tokens for ex in batch]
        purchase_sets = [ex.purchase_set for ex in batch]

        hist_tok = build_history_token_tensor(item_token_ids, histories).to(device)
        tgt_tok = build_target_query_tensor(vocab, targets, max_len=2).to(device)

        user_state = model.user_state(hist_tok)
        logits, _ = model.policy(user_state)
        actions = torch.stack([logit.argmax(dim=1) for logit in logits], dim=1)

        top_items = model.retrieve(actions, item_token_ids, topk=args.topk)
        top1 = top_items[:, 0].detach().cpu().numpy().tolist()

        mask = tgt_tok != pad
        q_em = (((actions == tgt_tok) | ~mask).all(dim=1)).float().detach().cpu().numpy().tolist()
        i_hit = [1.0 if int(top1[i]) in purchase_sets[i] else 0.0 for i in range(len(purchase_sets))]
        cons = [q_em[i] * i_hit[i] for i in range(len(i_hit))]

        q_em_list.extend(q_em)
        i_hit_list.extend(i_hit)
        cons_list.extend(cons)

    print(
        " ".join(
            [
                f"Q_EM={np.mean(q_em_list):.4f}",
                f"I_Hit@1={np.mean(i_hit_list):.4f}",
                f"Cons@1={np.mean(cons_list):.4f}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
