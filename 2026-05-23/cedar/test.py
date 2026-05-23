from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ToyEmbeddingDataset, build_factor_text_vocab
from model import CEDAR, reconstruction_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a * b).mean())


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")

    dim = int(ckpt["dim"])
    k = int(ckpt["k"])

    ds = ToyEmbeddingDataset(num_samples=4096, embedding_dim=dim, seed=123)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CEDAR(dim=dim, k=k).to(args.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.mean.copy_(ckpt["mean"].to(args.device))
    model._mean_initialized = True
    model.eval()

    total = 0.0
    n = 0
    active_ratios = []

    # Collect a matrix for simple axis-factor analysis
    z_sparse_list = []
    labels = defaultdict(list)

    with torch.no_grad():
        for batch in dl:
            z = batch["z"].to(args.device)
            out = model(z)
            loss = reconstruction_loss(out.z_hat, z, kind="l1")
            total += float(loss.cpu()) * z.shape[0]
            n += z.shape[0]
            active_ratios.append(float(out.active_mask.float().mean().cpu()))

            z_sparse_list.append(out.z_sparse.cpu().numpy())
            for key in ("color", "shape", "style"):
                labels[key].append(batch[key].numpy())

    z_sparse = np.concatenate(z_sparse_list, axis=0)
    for key in labels:
        labels[key] = np.concatenate(labels[key], axis=0)

    print(f"L1 reconstruction: {total / n:.6f}")
    print(f"avg active ratio: {np.mean(active_ratios):.4f} (target ~ k/D = {k/dim:.4f})")

    vocab = build_factor_text_vocab(ds.spec)
    print("\nAxis ↔ factor alignment (toy):")

    # For each axis, find the best-correlated factor value indicator
    for factor, names in vocab.items():
        print(f"\n[{factor}]")
        for axis in range(min(12, dim)):
            axis_values = z_sparse[:, axis]
            best = (None, -1.0)
            for value_id, name in enumerate(names):
                indicator = (labels[factor] == value_id).astype(np.float32)
                score = abs(corrcoef(axis_values, indicator))
                if score > best[1]:
                    best = (name, score)
            if best[0] is not None and best[1] > 0.15:
                print(f"axis {axis:02d}: {best[0]:<10s} |corr|={best[1]:.3f}")


if __name__ == "__main__":
    main()
