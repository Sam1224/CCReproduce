"""CMTA evaluation: compute simple ROC-AUC on the toy eval split."""
from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ToyVideoConfig, ToyVideoDataset
from model import CMTA, CMTAConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("cmta.infer")


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    for y in labels:
        if y == 1:
            tp += 1
        else:
            auc += tp * (1.0)  # rectangle width = 1 in rank space
            fp += 1
    auc = auc / (pos * neg) if pos * neg else 0.0
    return float(auc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/cmta.pt")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    state = torch.load(args.ckpt, map_location=args.device)
    cfg = CMTAConfig(**state["cfg"])
    model = CMTA(cfg).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()

    toy = ToyVideoConfig()
    eval_ds = ToyVideoDataset(toy, "eval")
    loader = DataLoader(eval_ds, batch_size=64)

    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            v = batch["v"].to(args.device)
            c = batch["c"].to(args.device)
            y = batch["y"].numpy().astype(np.int64)
            s = torch.sigmoid(model(v, c)).cpu().numpy()
            scores.extend(s.tolist())
            labels.extend(y.tolist())
    auc = roc_auc(np.array(scores), np.array(labels))
    logger.info("toy eval AUC = %.4f", auc)


if __name__ == "__main__":
    main()
