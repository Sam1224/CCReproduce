"""
Evaluation script for Dual-Path Content Moderation.
Primary metric from paper: Recall @ 80% Precision for each path.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(__file__))

from model.dual_path import DualPathModerationModel
from model.similarity_store import build_toy_violation_store
from data.livestream_dataset import ToyLivestreamDataset, collate_fn
from train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model = DualPathModerationModel(hidden_dim=256, feature_dim=128, embed_dim=64).to(args.device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        print(f"Loaded: {args.checkpoint}")
    else:
        print("Evaluating random init (toy demo).")

    violation_store = build_toy_violation_store(model, embed_dim=64)
    dataset = ToyLivestreamDataset(num_samples=args.num_samples, violation_rate=0.35, seed=99)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    metrics = evaluate(model, loader, violation_store, args.device)

    print("\n=== DualPathMod Evaluation Results ===")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Macro F1:            {metrics['macro_f1']:.4f}")
    print(f"Precision:           {metrics['precision']:.4f}")
    print(f"Recall:              {metrics['recall']:.4f}")
    print(f"Recall @ 80% Prec:   {metrics['recall_at_p80']:.4f}")
    print(f"\nPaper targets: Classification path 67%, Similarity path 76% recall@80%precision")


if __name__ == "__main__":
    main()
