from __future__ import annotations

import os
from pathlib import Path

import torch

from dataset import make_dataset, split
from model import MMClassifier


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def load_model(path: Path, hidden: int) -> MMClassifier:
    ckpt = torch.load(path, map_location="cpu")
    m = MMClassifier(hidden=hidden)
    m.load_state_dict(ckpt["state_dict"], strict=True)
    m.eval()
    return m


def main() -> None:
    batch = make_dataset(n=2000, seed=2)
    _, test = split(batch, 0.8)

    teacher = load_model(Path("checkpoints/teacher.pt"), hidden=256)
    vanilla = load_model(Path("checkpoints/student_vanilla.pt"), hidden=96)
    uakd = load_model(Path("checkpoints/student_uakd.pt"), hidden=96)

    with torch.no_grad():
        t = teacher(test.img, test.txt)
        v = vanilla(test.img, test.txt)
        u = uakd(test.img, test.txt)

    print(f"teacher_acc={accuracy(t, test.y):.3f}")
    print(f"vanilla_kd_acc={accuracy(v, test.y):.3f}")
    print(f"uakd_acc={accuracy(u, test.y):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
