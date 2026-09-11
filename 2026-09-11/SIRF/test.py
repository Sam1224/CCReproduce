from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SirfToyDataset, collate_sirf
from model import OneModel


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SirfToyDataset(num_samples=64, seed=123)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_sirf)
    model = OneModel().to(device)
    checkpoint = Path("checkpoints/sirf_toy.pt")
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    correct = 0
    total = 0
    black_scores = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        out = model(batch)
        probs = torch.softmax(out.verdict_logits, dim=-1)
        pred = probs.argmax(dim=-1)
        correct += int((pred == batch["verdict_label"]).sum().item())
        total += pred.numel()
        black_scores.extend(probs[:, 2].detach().cpu().tolist())
    print(f"verdict accuracy (toy): {correct / max(total, 1):.3f}")
    print(f"mean black risk score: {sum(black_scores) / max(len(black_scores), 1):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
