from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cq_sid.data.toy import (
    TextSeqDataset,
    build_toy_corpus,
    build_vocab,
    collate_text_batch,
)
from cq_sid.models.seq2seq import TinyTransformerSeq2Seq
from cq_sid.utils.format_sid import SID
from cq_sid.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=["query2sid", "user_query2sid"], default="query2sid")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="outputs/sft.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items, _users, examples = build_toy_corpus()

    # A simple CQ-SID design for the toy setting: category + (item-index, 0, 0)
    # In the original paper, IDs come from (category-aware) RQ-VAE quantization.
    item_to_sid_tokens = {}
    for idx, it in enumerate(items):
        sid = SID(category_id=it.category_id, codes=(idx, 0, 0))
        item_to_sid_tokens[it.item_id] = sid.to_tokens()

    queries = [ex.query for ex in examples]
    user_profiles = [ex.user_profile for ex in examples]
    targets = [item_to_sid_tokens[ex.item_id] for ex in examples]

    query_vocab = build_vocab(
        texts=[f"{u} {q}" for u, q in zip(user_profiles, queries)], extra_tokens=[]
    )
    sid_tokens = [t for toks in targets for t in toks]
    target_vocab = build_vocab(texts=[], extra_tokens=sid_tokens)

    dataset = TextSeqDataset(
        queries=queries,
        targets=targets,
        query_vocab=query_vocab,
        target_vocab=target_vocab,
        add_user_profile=args.stage == "user_query2sid",
        user_profiles=user_profiles,
    )

    pad_q = query_vocab.token_to_id["<pad>"]
    pad_t = target_vocab.token_to_id["<pad>"]
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=True,
        collate_fn=lambda b: collate_text_batch(b, pad_q, pad_t),
    )

    model = TinyTransformerSeq2Seq(
        src_vocab_size=len(query_vocab.id_to_token),
        tgt_vocab_size=len(target_vocab.id_to_token),
        d_model=192,
        nhead=4,
        num_layers=2,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            src_ids = batch["query_ids"].to(device)
            src_mask = batch["query_mask"].to(device)
            tgt_ids = batch["target_ids"].to(device)
            tgt_mask = batch["target_mask"].to(device)
            out = model(src_ids, src_mask, tgt_ids, tgt_mask, pad_id=pad_t)
            opt.zero_grad()
            out.loss.backward()
            opt.step()

        if epoch % 50 == 0:
            print(f"epoch={epoch} loss={out.loss.item():.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": args.stage,
            "query_vocab": query_vocab.to_state(),
            "target_vocab": target_vocab.to_state(),
            "model": model.state_dict(),
        },
        args.out,
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
