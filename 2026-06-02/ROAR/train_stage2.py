from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from data import build_toy_ecommerce_dataset
from evaluate import compute_metrics_for_groups
from losses import ROARBatch, contrastive_loss_with_false_negative_mask, roar_alignment_loss
from model import DualEncoder, SimpleTokenizer


def _collate(tokenizer: SimpleTokenizer, batch) -> tuple:
    queries = [ex.query for ex in batch]
    flat_products: List[str] = []
    group_sizes: List[int] = []
    for ex in batch:
        group_sizes.append(len(ex.products))
        flat_products.extend(ex.products)

    q_batch = tokenizer.encode_batch(queries)
    d_batch = tokenizer.encode_batch(flat_products)
    return q_batch, d_batch, ROARBatch(group_sizes=group_sizes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=str, default="runs/stage1.pt")
    parser.add_argument("--out", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = build_toy_ecommerce_dataset()

    # Load stage1
    stage1 = torch.load(args.stage1, map_location="cpu")
    tokenizer = SimpleTokenizer()
    tokenizer.vocab = stage1["tokenizer_vocab"]

    model = DualEncoder(vocab_size=len(tokenizer.vocab), hidden_size=stage1["config"]["hidden_size"]).to(device)
    model.load_state_dict(stage1["model_state"], strict=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    dl = DataLoader(
        bundle.stage2_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(tokenizer, b),
        drop_last=True,
    )

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for q_batch, d_batch, roar_batch in dl:
            q_batch.token_ids = q_batch.token_ids.to(device)
            q_batch.attn_mask = q_batch.attn_mask.to(device)
            d_batch.token_ids = d_batch.token_ids.to(device)
            d_batch.attn_mask = d_batch.attn_mask.to(device)

            q_emb = model.encode(q_batch.token_ids, q_batch.attn_mask)
            d_emb = model.encode(d_batch.token_ids, d_batch.attn_mask)

            # Contrastive term: treat the first product of each group as positive.
            # We realize this by slicing the flattened doc tensor.
            offsets = []
            s = 0
            for g in roar_batch.group_sizes:
                offsets.append(s)
                s += g
            pos_docs = d_emb[torch.tensor(offsets, device=device)]
            l_contrast = contrastive_loss_with_false_negative_mask(q_emb, pos_docs)
            l_align = roar_alignment_loss(q_emb, d_emb, roar_batch)
            loss = l_contrast + args.beta * l_align

            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.detach().cpu())

        print(f"epoch={epoch} loss={total/len(dl):.4f}")

        # quick eval
        eval_groups = [(g.query, g.products) for g in bundle.eval_groups[:200]]
        m = compute_metrics_for_groups(model, tokenizer, device, eval_groups)
        print(f"eval map@8={m.map_at_8:.4f} ndcg@8={m.ndcg_at_8:.4f} auc={m.auc:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "tokenizer_vocab": tokenizer.vocab,
        "model_state": model.state_dict(),
        "config": {"hidden_size": stage1["config"]["hidden_size"]},
        "beta": args.beta,
    }
    out_path = out_dir / "stage2_roar.pt"
    torch.save(ckpt, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
