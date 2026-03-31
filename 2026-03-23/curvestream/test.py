from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CurveStreamToyDataset, collate
from model import CurveStreamConfig, CurveStreamToyModel


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)

    ckpt_path = Path("checkpoints/curvestream.pt")
    if not ckpt_path.exists():
        raise SystemExit("missing checkpoint: run train.py first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = CurveStreamConfig(**ckpt["cfg"])
    model = CurveStreamToyModel(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = CurveStreamToyDataset(n_samples=1000, seed=999)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)

    total = 0
    correct = 0
    mem_tokens = 0.0

    with torch.no_grad():
        for batch in loader:
            frames = batch.frames.to(device)
            labels = batch.labels.to(device)
            logits, stats = model(frames)
            pred = logits.argmax(dim=-1)
            total += int(labels.numel())
            correct += int((pred == labels).sum().item())
            mem_tokens += float(stats["mem_tokens"].float().mean().item())

    print(f"acc={correct/max(1,total):.4f} avg_mem_tokens={mem_tokens/max(1,len(loader)):.2f} policy={cfg.policy}")


if __name__ == "__main__":
    main()
