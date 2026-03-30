from __future__ import annotations

import os
from pathlib import Path

import torch

from dataset import make_dataset
from model import FusionModel


def eval_acc(model: FusionModel, batch) -> float:
    with torch.no_grad():
        pred = model(batch.img, batch.q).argmax(dim=-1)
    return (pred == batch.y).float().mean().item()


def main() -> None:
    ckpt = torch.load("checkpoints/vlm_realworld.pt", map_location="cpu")
    model = FusionModel(img_dim=int(ckpt["img_dim"]), q_vocab=4, num_answers=6)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    id_test = make_dataset(n=1500, seed=2, ood=False)
    ood_test = make_dataset(n=1500, seed=3, ood=True)

    print(f"in_domain_acc={eval_acc(model, id_test):.3f}")
    print(f"ood_acc={eval_acc(model, ood_test):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
