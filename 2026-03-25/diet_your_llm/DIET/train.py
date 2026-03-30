from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyTaskDataset, build_vocab, collate_batch
from model import TinyLMConfig, TinyTransformerLM, lm_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = build_vocab()
    pad_id = vocab.stoi["<pad>"]

    tasks = [
        ToyTaskDataset("add", vocab=vocab, num_samples=512, seed=0),
        ToyTaskDataset("parity", vocab=vocab, num_samples=512, seed=1),
        ToyTaskDataset("compare", vocab=vocab, num_samples=512, seed=2),
    ]

    ds = torch.utils.data.ConcatDataset(tasks)
    dl = DataLoader(
        ds,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id=pad_id),
    )

    cfg = TinyLMConfig(vocab_size=len(vocab.tokens), d_model=128, nhead=4, nlayer=2, d_ff=256, max_len=32)
    model = TinyTransformerLM(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    model.train()
    for epoch in range(3):
        for step, batch in enumerate(dl):
            logits, _ = model(input_ids=batch["input_ids"].to(device), attn_mask=batch["attn_mask"].to(device))
            loss = lm_loss(logits, batch["target_id"].to(device))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if step % 50 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(), "vocab": vocab.tokens}, ckpt_dir / "base.pt")
    print(f"Saved: {ckpt_dir / 'base.pt'}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
