from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sage.dataset import make_toy_pu_dataset
from sage.metrics import compute_binary_metrics
from sage.model import MLPClassifier


def _load_model(ckpt_path: Path, device: str) -> MLPClassifier:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MLPClassifier(n_features=int(ckpt["n_features"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = make_toy_pu_dataset(seed=args.seed)
    n = data.x.shape[0]

    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=rng)
    n_train = int(n * 0.7)
    test_idx = perm[n_train:]

    x_test = data.x[test_idx].float()
    y_test = data.y_true[test_idx]
    edge_test = data.edge_cohort[test_idx]

    model = _load_model(Path(args.ckpt), args.device)

    with torch.no_grad():
        logits = model(x_test.to(args.device)).cpu()
        y_pred = (torch.sigmoid(logits) >= 0.5).to(torch.int64)

    metrics = compute_binary_metrics(y_true=y_test, y_pred=y_pred, edge_cohort=edge_test)
    result = {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "edge_fpr": metrics.edge_fpr,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
