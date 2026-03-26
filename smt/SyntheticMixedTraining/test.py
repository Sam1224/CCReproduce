from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SyntheticMixedToyDataset, collate_smt
from model import OneModel


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SyntheticMixedToyDataset(num_samples=64, vocab_size=4000)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_smt)

    model = OneModel(vocab_size=4000, d_model=256).to(device)
    model.eval()

    qa_loss = 0.0
    doc_loss = 0.0
    n = 0

    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)

        qa_ce = torch.nn.functional.cross_entropy(
            out.qa_logits.reshape(-1, out.qa_logits.shape[-1]), batch["qa_tgt"].reshape(-1), ignore_index=-100
        )
        doc_ce = torch.nn.functional.cross_entropy(
            out.doc_logits.reshape(-1, out.doc_logits.shape[-1]), batch["doc_tgt"].reshape(-1), ignore_index=-100
        )

        qa_loss += float(qa_ce)
        doc_loss += float(doc_ce)
        n += 1

    print(f"QA CE (toy): {qa_loss / max(n, 1):.3f}")
    print(f"Doc CE (toy): {doc_loss / max(n, 1):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
