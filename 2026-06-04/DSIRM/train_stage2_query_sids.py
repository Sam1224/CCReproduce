from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import build_toy_dataset
from model import DualEncoder, QuerySidPredictor, SimpleTokenizer
from rqvae import ResidualVectorQuantizer


def _collate(tokenizer: SimpleTokenizer, batch):
    queries = [ex.query for ex in batch]
    items = [ex.item for ex in batch]
    q_batch = tokenizer.encode_batch(queries)
    d_batch = tokenizer.encode_batch(items)
    return q_batch, d_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=str, default="runs/stage1.pt")
    parser.add_argument("--out", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1 = torch.load(args.stage1, map_location="cpu")

    cfg = stage1["config"]
    hidden = int(cfg["hidden"])
    num_codebooks = int(cfg["codebooks"])
    codebook_size = int(cfg["codebook_size"])

    tokenizer = SimpleTokenizer()
    tokenizer.vocab = stage1["tokenizer_vocab"]

    encoder = DualEncoder(vocab_size=len(tokenizer.vocab), hidden_size=hidden)
    encoder.load_state_dict(stage1["encoder_state"])
    encoder.to(device)
    encoder.eval()

    quantizer = ResidualVectorQuantizer(dim=hidden, num_codebooks=num_codebooks, codebook_size=codebook_size)
    quantizer.load_state_dict(stage1["quantizer_state"])
    quantizer.to(device)
    quantizer.eval()

    predictor = QuerySidPredictor(
        vocab_size=len(tokenizer.vocab),
        hidden_size=hidden,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
    ).to(device)

    optim = torch.optim.AdamW(predictor.parameters(), lr=args.lr)

    bundle = build_toy_dataset()

    dl = DataLoader(
        bundle.stage1_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(tokenizer, b),
        drop_last=True,
    )

    predictor.train()
    for epoch in range(args.epochs):
        total = 0.0
        for q_batch, d_batch in dl:
            q_batch.token_ids = q_batch.token_ids.to(device)
            q_batch.attn_mask = q_batch.attn_mask.to(device)
            d_batch.token_ids = d_batch.token_ids.to(device)
            d_batch.attn_mask = d_batch.attn_mask.to(device)

            with torch.no_grad():
                item_emb = encoder.encode(d_batch.token_ids, d_batch.attn_mask)
                target_codes = quantizer(item_emb).codes  # (B, L)

            logits = predictor(q_batch)  # (B, L, K)
            loss = 0.0
            for level in range(num_codebooks):
                loss = loss + torch.nn.functional.cross_entropy(logits[:, level, :], target_codes[:, level].to(device))

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += float(loss.detach().cpu())

        print(f"epoch={epoch} loss={total/len(dl):.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "predictor_state": predictor.state_dict(),
        "config": {
            "hidden": hidden,
            "codebooks": num_codebooks,
            "codebook_size": codebook_size,
        },
    }

    out_path = out_dir / "stage2.pt"
    torch.save(ckpt, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
