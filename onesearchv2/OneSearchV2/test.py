from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import OneSearchV2ToyDataset, collate_onesearchv2
from model import OneModel


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = OneSearchV2ToyDataset(num_samples=64, vocab_size=5000, cot_len=8)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_onesearchv2)

    model = OneModel(vocab_size=5000, d_model=256, cot_len=8).to(device)
    model.eval()

    total = 0
    correct = 0

    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)

        pred = out.answer_logits.argmax(dim=-1)
        tgt = batch["answer_tgt"]
        mask = tgt != -100
        correct += int((pred[mask] == tgt[mask]).sum().item())
        total += int(mask.sum().item())

    print(f"Next-token accuracy (toy): {correct / max(total, 1):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
