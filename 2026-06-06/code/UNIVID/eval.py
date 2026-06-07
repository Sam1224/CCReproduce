"""
UNIVID Evaluation Script
Metrics:
    - Violation Leakage Rate: fraction of violations missed (FN / (TP + FN))
    - Overkill Rate: fraction of clean videos incorrectly flagged (FP / (TN + FP))
    - Precision, Recall, F1 per category

Usage:
    python eval.py --batch_size 8 --threshold 0.5
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from model.univid import UNIVID
from pipeline.moderation_actor import ModerationActor
from data.dataset import ModerationDataset


def evaluate(backbone: UNIVID, actor: ModerationActor, dataloader: DataLoader,
             threshold: float, device: str):
    backbone.eval()
    actor.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device)
            policy_ids = batch["policy_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = backbone(frames, policy_ids)
            caption_emb = outputs["caption_embedding"]
            actor_out = actor(caption_emb)
            pred = actor_out["combined_scores"]

            # Align dimensions
            C = pred.shape[1]
            L = labels.shape[1]
            if L < C:
                pad = torch.zeros(labels.shape[0], C - L, device=device)
                labels = torch.cat([labels, pad], dim=1)
            elif L > C:
                labels = labels[:, :C]

            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()   # (N, C)
    labels = torch.cat(all_labels, dim=0).numpy()  # (N, C)

    binary_preds = (preds >= threshold).astype(float)

    # Any-violation level (video-level)
    pred_any = binary_preds.max(axis=1)
    label_any = labels.max(axis=1)

    TP = ((pred_any == 1) & (label_any == 1)).sum()
    FP = ((pred_any == 1) & (label_any == 0)).sum()
    TN = ((pred_any == 0) & (label_any == 0)).sum()
    FN = ((pred_any == 0) & (label_any == 1)).sum()

    leakage = FN / (TP + FN + 1e-8)
    overkill = FP / (TN + FP + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\n=== UNIVID Evaluation (threshold={threshold}) ===")
    print(f"  Violation Leakage Rate : {leakage:.4f}  (lower is better)")
    print(f"  Overkill Rate          : {overkill:.4f}  (lower is better)")
    print(f"  Precision              : {precision:.4f}")
    print(f"  Recall                 : {recall:.4f}")
    print(f"  F1                     : {f1:.4f}")
    print(f"  TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    print()

    return {
        "leakage": float(leakage),
        "overkill": float(overkill),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    dataset = ModerationDataset(size=args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    backbone = UNIVID()
    actor = ModerationActor(embed_dim=256, num_categories=64)

    evaluate(backbone, actor, dataloader, args.threshold, args.device)


if __name__ == "__main__":
    main()
