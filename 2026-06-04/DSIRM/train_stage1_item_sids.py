from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import build_toy_dataset
from model import DualEncoder, SimpleTokenizer
from rqvae import ResidualVectorQuantizer, rq_contrastive_loss


def _collate(tokenizer: SimpleTokenizer, batch):
    queries = [ex.query for ex in batch]
    items = [ex.item for ex in batch]
    q_batch = tokenizer.encode_batch(queries)
    d_batch = tokenizer.encode_batch(items)
    return q_batch, d_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--codebooks", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = build_toy_dataset()
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([ex.query for ex in bundle.stage1_train] + [ex.item for ex in bundle.stage1_train])

    encoder = DualEncoder(vocab_size=len(tokenizer.vocab), hidden_size=args.hidden).to(device)
    quantizer = ResidualVectorQuantizer(dim=args.hidden, num_codebooks=args.codebooks, codebook_size=args.codebook_size).to(
        device
    )

    optim = torch.optim.AdamW(list(encoder.parameters()) + list(quantizer.parameters()), lr=args.lr)

    dl = DataLoader(
        bundle.stage1_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(tokenizer, b),
        drop_last=True,
    )

    encoder.train()
    quantizer.train()

    for epoch in range(args.epochs):
        total = 0.0
        for q_batch, d_batch in dl:
            q_batch.token_ids = q_batch.token_ids.to(device)
            q_batch.attn_mask = q_batch.attn_mask.to(device)
            d_batch.token_ids = d_batch.token_ids.to(device)
            d_batch.attn_mask = d_batch.attn_mask.to(device)

            q, d = encoder(q_batch, d_batch)  # (B,H)
            q_out = quantizer(d)

            loss = rq_contrastive_loss(
                q=q,
                d_pre=d,
                d_quant=q_out.quantized,
                codebook_loss=q_out.codebook_loss,
                commitment_loss=q_out.commitment_loss,
                beta=args.beta,
                temperature=args.temperature,
            )

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += float(loss.detach().cpu())

        print(f"epoch={epoch} loss={total/len(dl):.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "tokenizer_vocab": tokenizer.vocab,
        "encoder_state": encoder.state_dict(),
        "quantizer_state": quantizer.state_dict(),
        "config": {
            "hidden": args.hidden,
            "codebooks": args.codebooks,
            "codebook_size": args.codebook_size,
        },
    }

    out_path = out_dir / "stage1.pt"
    torch.save(ckpt, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
