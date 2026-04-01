from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import TEMPLATE_POOL, iter_vocab, load_toy_dataset, tokenize
from model import ReLogClassifier


def build_vocab_index(vocab: list[str]) -> dict[str, int]:
    return {token: idx for idx, token in enumerate(vocab)}


def encode(text: str, *, vocab_index: dict[str, int], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenize(text)
    ids = [vocab_index.get(t, vocab_index.get("<unk>", 1)) for t in tokens][:max_len]
    attn = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(vocab_index.get("<pad>", 0))
        attn.append(0)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.float32)


def collate(batch, *, vocab_index: dict[str, int], max_len: int):
    token_ids_list = []
    attn_list = []
    level_list = []
    template_list = []

    for example in batch:
        text = example.code + "\n" + example.runtime_trace
        token_ids, attn = encode(text, vocab_index=vocab_index, max_len=max_len)
        token_ids_list.append(token_ids)
        attn_list.append(attn)
        level_list.append(int(example.target_level))
        template_list.append(TEMPLATE_POOL.index(example.target_template))

    return (
        torch.stack(token_ids_list, dim=0),
        torch.stack(attn_list, dim=0),
        torch.tensor(level_list, dtype=torch.long),
        torch.tensor(template_list, dtype=torch.long),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--out", default="ckpt.pt")
    args = ap.parse_args()

    examples = load_toy_dataset()
    vocab = iter_vocab(examples)
    vocab_index = build_vocab_index(vocab)

    model = ReLogClassifier(vocab_size=len(vocab), num_levels=4, num_templates=len(TEMPLATE_POOL))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    loader = DataLoader(
        examples,
        batch_size=min(8, len(examples)),
        shuffle=True,
        collate_fn=lambda b: collate(b, vocab_index=vocab_index, max_len=args.max_len),
    )

    model.train()
    for _ in range(args.epochs):
        for token_ids, attn, level_target, template_target in loader:
            out = model(token_ids, attn)
            loss = loss_fn(out.level_logits, level_target) + loss_fn(out.template_logits, template_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    out_path = Path(args.out)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "max_len": args.max_len,
        },
        out_path,
    )
    print(f"saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
