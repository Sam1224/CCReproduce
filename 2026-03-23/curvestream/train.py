from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CurveStreamToyDataset, collate
from model import CurveStreamConfig, CurveStreamToyModel


def evaluate(model: CurveStreamToyModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    mem_tokens = 0.0
    long_tokens = 0.0

    with torch.no_grad():
        for batch in loader:
            frames = batch.frames.to(device)
            labels = batch.labels.to(device)
            logits, stats = model(frames)
            pred = logits.argmax(dim=-1)
            total += int(labels.numel())
            correct += int((pred == labels).sum().item())
            mem_tokens += float(stats["mem_tokens"].float().mean().item())
            long_tokens += float(stats["long_tokens"].float().mean().item())

    return {
        "acc": correct / max(1, total),
        "avg_mem_tokens": mem_tokens / max(1, len(loader)),
        "avg_long_tokens": long_tokens / max(1, len(loader)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--policy", type=str, default="curvature", choices=["curvature", "uniform", "recent"])
    parser.add_argument("--mem-long", type=int, default=8)
    parser.add_argument("--mem-short", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="checkpoints/curvestream.pt")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    full = CurveStreamToyDataset(n_samples=8000, seed=args.seed)
    train = CurveStreamToyDataset(n_samples=6500, seed=args.seed)
    val = CurveStreamToyDataset(n_samples=1500, seed=args.seed + 123)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    cfg = CurveStreamConfig(
        num_classes=full.num_classes,
        dim=full.dim,
        hidden=64,
        mem_long=args.mem_long,
        mem_short=args.mem_short,
        policy=args.policy,
    )
    model = CurveStreamToyModel(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            frames = batch.frames.to(device)
            labels = batch.labels.to(device)

            logits, _ = model(frames)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        metrics = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch:02d} loss={loss.item():.4f} val_acc={metrics['acc']:.4f} "
            f"avg_mem={metrics['avg_mem_tokens']:.2f} avg_long={metrics['avg_long_tokens']:.2f} policy={args.policy}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": cfg.__dict__,
            "state_dict": model.state_dict(),
        },
        out_path,
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
