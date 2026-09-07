from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import ToyTradeUpDataset, split_dataset
from model import TradeUpStudent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/tradeup_student.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ToyTradeUpDataset(num_samples=2048)
    _, test_set = split_dataset(dataset)
    model = TradeUpStudent(num_product_types=dataset.num_product_types).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    scores, labels = [], []
    with torch.no_grad():
        for batch in DataLoader(test_set, batch_size=args.batch_size):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(batch)
            scores.extend(output.tradeup_score.cpu().tolist())
            labels.extend(batch["binary_label"].cpu().tolist())
    pred = (np.asarray(scores) >= 0.5).astype(int)
    print({
        "auc": round(float(roc_auc_score(labels, scores)), 4),
        "ap": round(float(average_precision_score(labels, scores)), 4),
        "f1": round(float(f1_score(labels, pred)), 4),
    })


if __name__ == "__main__":
    main()
