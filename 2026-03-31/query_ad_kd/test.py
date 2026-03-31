from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import QueryAdDataset, collate
from model import StudentTwoTower
from train import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    ds = QueryAdDataset(n=2000, seed=17)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.ckpt, map_location=device)
    student = StudentTwoTower().to(device)
    student.load_state_dict(payload["student"], strict=True)

    metrics = evaluate(student, loader, device)
    print(metrics)


if __name__ == "__main__":
    main()
