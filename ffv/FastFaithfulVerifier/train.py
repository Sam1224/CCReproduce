from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import VerifierToyDataset, collate_verifier
from model import OneModel, verifier_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed_everything(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VerifierToyDataset(num_samples=256, vocab_size=5000, min_len=128, max_len=256)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_verifier)

    model = OneModel(vocab_size=5000, d_model=256, num_layers=8, early_exit_threshold=0.9).to(device)
    optim = model.get_optim(lr=3e-4)

    model.train()
    for epoch in range(1):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            loss, logs = verifier_loss(
                out,
                span_labels=batch["span_labels"],
                halluc=batch["halluc"],
                attn_mask=batch["attn_mask"],
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if step % 20 == 0:
                print(
                    f"epoch={epoch} step={step} total={loss.item():.4f} "
                    f"span={logs['span_loss']:.4f} exit={logs['exit_loss']:.4f}"
                )


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
