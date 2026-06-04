from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import build_toy_dataset
from model import DualEncoder, QuerySidPredictor, SimpleTokenizer
from ranker import SidRanker
from rqvae import ResidualVectorQuantizer
from sid import prefix_match_features


def _collate(tokenizer: SimpleTokenizer, batch):
    queries = [ex.query for ex in batch]
    items = [ex.item for ex in batch]
    labels = torch.tensor([ex.label for ex in batch], dtype=torch.float32)
    q_batch = tokenizer.encode_batch(queries)
    d_batch = tokenizer.encode_batch(items)
    return q_batch, d_batch, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=str, default="runs/stage1.pt")
    parser.add_argument("--stage2", type=str, default="runs/stage2.pt")
    parser.add_argument("--out", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1 = torch.load(args.stage1, map_location="cpu")
    stage2 = torch.load(args.stage2, map_location="cpu")

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
    )
    predictor.load_state_dict(stage2["predictor_state"])
    predictor.to(device)
    predictor.eval()

    ranker = SidRanker(num_codebooks=num_codebooks).to(device)
    optim = torch.optim.AdamW(ranker.parameters(), lr=args.lr)

    bundle = build_toy_dataset()
    dl = DataLoader(
        bundle.rank_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(tokenizer, b),
        drop_last=True,
    )

    ranker.train()
    for epoch in range(args.epochs):
        total = 0.0
        for q_batch, d_batch, y in dl:
            q_batch.token_ids = q_batch.token_ids.to(device)
            q_batch.attn_mask = q_batch.attn_mask.to(device)
            d_batch.token_ids = d_batch.token_ids.to(device)
            d_batch.attn_mask = d_batch.attn_mask.to(device)
            y = y.to(device)

            with torch.no_grad():
                q_emb = encoder.encode(q_batch.token_ids, q_batch.attn_mask)
                d_emb = encoder.encode(d_batch.token_ids, d_batch.attn_mask)
                d_codes = quantizer(d_emb).codes

                q_logits = predictor(q_batch)
                q_codes = torch.argmax(q_logits, dim=-1)

                dense_sim = torch.sum(q_emb * d_emb, dim=-1, keepdim=True)
                prefix = prefix_match_features(q_codes, d_codes)

                feats = torch.cat([dense_sim, prefix], dim=1)

            logits = ranker(feats)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += float(loss.detach().cpu())

        print(f"epoch={epoch} loss={total/len(dl):.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "ranker_state": ranker.state_dict(),
        "config": {"num_codebooks": num_codebooks},
    }

    out_path = out_dir / "stage3.pt"
    torch.save(ckpt, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
