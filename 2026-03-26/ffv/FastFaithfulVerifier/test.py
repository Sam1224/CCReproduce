from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import VerifierToyDataset, collate_verifier
from model import OneModel


def _f1(tp: int, fp: int, fn: int) -> float:
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-12)


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VerifierToyDataset(num_samples=64, vocab_size=5000, min_len=128, max_len=256)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_verifier)

    model = OneModel(vocab_size=5000, d_model=256, num_layers=8, early_exit_threshold=0.9).to(device)
    model.eval()

    tp = fp = fn = 0
    exit_layers = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)

        span_pred = (torch.sigmoid(out.span_logits) > 0.5)
        span_gt = batch["span_labels"] > 0.5
        mask = batch["attn_mask"]

        tp += int(((span_pred & span_gt) & mask).sum().item())
        fp += int(((span_pred & ~span_gt) & mask).sum().item())
        fn += int(((~span_pred & span_gt) & mask).sum().item())

        exit_layers.append(int(out.exit_layer.item()))

    print(f"Unsupported-span token F1 (toy): {_f1(tp, fp, fn):.3f}")
    print(f"Average early-exit layer (toy, 0-index): {sum(exit_layers) / max(len(exit_layers), 1):.2f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
