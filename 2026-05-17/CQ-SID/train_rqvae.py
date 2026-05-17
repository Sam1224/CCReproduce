from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cq_sid.data.toy import build_toy_corpus, build_vocab
from cq_sid.models.rqvae import RQVAE
from cq_sid.models.text_encoder import MeanPoolTextEncoder
from cq_sid.utils.seed import set_seed
from cq_sid.utils.tokenizer import basic_tokenize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/rqvae.pt")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items, _users, _examples = build_toy_corpus()
    texts = [it.title for it in items]

    vocab = build_vocab(texts, extra_tokens=[])
    encoder = MeanPoolTextEncoder(vocab_size=len(vocab.id_to_token), emb_dim=128).to(device)
    model = RQVAE(input_dim=128, latent_dim=64, num_codebooks=3, codebook_size=64).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(model.parameters()), lr=args.lr)

    tokenized = [basic_tokenize(t) for t in texts]
    token_ids = [torch.tensor(vocab.encode(toks), dtype=torch.long) for toks in tokenized]

    def collate(batch):
        max_len = max(x.numel() for x in batch)
        pad_id = vocab.token_to_id["<pad>"]
        out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for i, x in enumerate(batch):
            out[i, : x.numel()] = x
            mask[i, : x.numel()] = True
        return out.to(device), mask.to(device)

    loader = DataLoader(token_ids, batch_size=len(items), shuffle=True, collate_fn=collate)

    for epoch in range(1, args.epochs + 1):
        for ids, mask in loader:
            x = encoder(ids, mask)
            out = model(x)
            loss = out.loss_recon + out.loss_commit
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print(f"epoch={epoch} loss={loss.item():.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"vocab": vocab.to_state(), "encoder": encoder.state_dict(), "rqvae": model.state_dict()},
        args.out,
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
