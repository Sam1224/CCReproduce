from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SyntheticMixedToyDataset, collate_smt
from model import OneModel, smt_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SyntheticMixedToyDataset(num_samples=256, vocab_size=4000)
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_smt)

    model = OneModel(vocab_size=4000, d_model=256).to(device)
    optim = model.get_optim(lr=3e-4)

    model.train()
    for epoch in range(1):
        for step, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            loss, logs = smt_loss(out, qa_tgt=batch["qa_tgt"], doc_tgt=batch["doc_tgt"], qa_weight=1.0, doc_weight=1.0)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if step % 20 == 0:
                print(
                    f"epoch={epoch} step={step} total={loss.item():.4f} "
                    f"qa={logs['qa_loss']:.4f} doc={logs['doc_loss']:.4f}"
                )


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
