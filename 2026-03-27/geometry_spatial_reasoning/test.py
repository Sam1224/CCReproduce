from __future__ import annotations

import os
from pathlib import Path

import torch

from dataset import make_dataset
from model import BaselineMLP, GeometryAware, acc


def main() -> None:
    ckpt = torch.load("checkpoints/geometry.pt", map_location="cpu")
    base = BaselineMLP(); geo = GeometryAware()
    base.load_state_dict(ckpt["baseline"], strict=True)
    geo.load_state_dict(ckpt["geometry"], strict=True)
    base.eval(); geo.eval()

    test = make_dataset(n=2000, seed=2)
    with torch.no_grad():
        a0 = acc(base(test.pts, test.q), test.y)
        a1 = acc(geo(test.pts, test.q), test.y)

    print(f"baseline_acc={a0:.3f}")
    print(f"geometry_acc={a1:.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
