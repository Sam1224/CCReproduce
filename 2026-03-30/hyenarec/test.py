from __future__ import annotations

import os
from pathlib import Path

import torch

from dataset import make_dataset
from model import HyenaSeqRec


def hit_at_k(logits: torch.Tensor, y: torch.Tensor, k: int = 10) -> float:
    topk = torch.topk(logits, k=k, dim=-1).indices
    hit = (topk == y.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return hit


def main() -> None:
    ckpt = torch.load("checkpoints/hyenarec.pt", map_location="cpu")
    model = HyenaSeqRec()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    test = make_dataset(n=3000, seed=2)
    with torch.no_grad():
        logits = model(test.seq)
    print(f"Hit@10={hit_at_k(logits, test.y, 10):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
