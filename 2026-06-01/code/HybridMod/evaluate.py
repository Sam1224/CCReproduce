"""
Evaluation script for HybridMod.
Computes Recall @ fixed Precision thresholds (as reported in paper).

Paper reports:
  Classification pipeline: 67% Recall @ 80% Precision
  Similarity pipeline:     76% Recall @ 80% Precision
"""
import argparse
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve

from model import HybridMod
from dataset import get_dataloaders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--target_precision", type=float, default=0.80)
    return p.parse_args()


def recall_at_precision(y_true, y_score, target_precision: float = 0.80):
    """Compute Recall at a fixed Precision operating point."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # Find threshold where precision >= target
    valid = precision >= target_precision
    if not valid.any():
        return 0.0, 0.0
    idx = np.where(valid)[0][0]
    return float(recall[idx]), float(precision[idx])


@torch.no_grad()
def collect_scores(model, loader, device):
    model.eval()
    all_labels = []
    all_scores_a = []  # Pipeline A violation probability
    all_scores_b = []  # Pipeline B violation probability
    all_scores_fused = []

    for batch in loader:
        text = batch["text_feat"].to(device)
        audio = batch["audio_feat"].to(device)
        visual = batch["visual_feat"].to(device)
        is_violation = batch["is_violation"].cpu().numpy()

        out = model(text, audio, visual)
        # Violation probability = 1 - P(non-violation) if class 0 is clean
        prob_a = torch.softmax(out["logits_a"], dim=-1)[:, 1:].sum(-1).cpu().numpy()
        prob_b = torch.softmax(out["logits_b"], dim=-1)[:, 1:].sum(-1).cpu().numpy()
        prob_fused = torch.softmax(out["logits"], dim=-1)[:, 1:].sum(-1).cpu().numpy()

        all_labels.append(is_violation)
        all_scores_a.append(prob_a)
        all_scores_b.append(prob_b)
        all_scores_fused.append(prob_fused)

    y_true = np.concatenate(all_labels)
    scores_a = np.concatenate(all_scores_a)
    scores_b = np.concatenate(all_scores_b)
    scores_fused = np.concatenate(all_scores_fused)
    return y_true, scores_a, scores_b, scores_fused


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(args.batch_size, args.num_classes)

    model = HybridMod(
        text_dim=768, audio_dim=128, visual_dim=2048,
        hidden_dim=512, num_classes=args.num_classes
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    y_true, scores_a, scores_b, scores_fused = collect_scores(model, test_loader, device)

    tp = args.target_precision
    recall_a, prec_a = recall_at_precision(y_true, scores_a, tp)
    recall_b, prec_b = recall_at_precision(y_true, scores_b, tp)
    recall_f, prec_f = recall_at_precision(y_true, scores_fused, tp)

    print(f"\n=== HybridMod Evaluation ===")
    print(f"Target Precision: {tp:.0%}")
    print(f"Pipeline A (Classifier):  Recall={recall_a:.4f}, Precision={prec_a:.4f}")
    print(f"Pipeline B (Similarity):  Recall={recall_b:.4f}, Precision={prec_b:.4f}")
    print(f"Fused:                    Recall={recall_f:.4f}, Precision={prec_f:.4f}")
    print("\n(Paper reports: Classifier 67%, Similarity 76% recall @ 80% precision)")


if __name__ == "__main__":
    main()
