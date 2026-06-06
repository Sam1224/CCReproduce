"""
UNIVID evaluation script.

Reports the three metrics central to UNIVID's production evaluation:
  - Violation F1      : harmonic mean of precision and recall on violation detection
  - Leakage Rate      : FN / (FN + TP)  — violating content that passes through
  - Overkill Rate     : FP / (FP + TN)  — benign content incorrectly blocked

These map directly to the paper's stated improvements:
  "reduces violation leakage by 42.7% and overkill rate by 37.0% relatively"  [UNIVID §4]

Also reports per-policy breakdown and confusion matrix.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix, classification_report
)

from data import UNIVIDDataset, generate_dataset, POLICIES
from model import UNIVID


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load dataset ──
    if os.path.exists(args.data):
        raw = torch.load(args.data)
        dataset = UNIVIDDataset(raw)
    else:
        print("No dataset file found; generating evaluation data …")
        dataset = generate_dataset(n_videos=300)

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # ── Load model ──
    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNIVID(frame_dim=64, hidden_dim=128, n_heads=4, max_seq=12).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_preds = []
    all_labels = []
    all_policy_ids = []

    with torch.no_grad():
        for batch in loader:
            video_feat  = batch["video_feat"].to(device)
            policy_id   = batch["policy_id"].to(device)
            caption_ids = batch["caption_ids"].to(device)
            violation   = batch["violation"]

            preds = model.predict_violation(video_feat, policy_id, caption_ids)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(violation.long().tolist())
            all_policy_ids.extend(policy_id.cpu().tolist())

    y_pred = all_preds
    y_true = all_labels

    # ── Global metrics ──
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, sum(y_true))

    f1       = f1_score(y_true, y_pred, zero_division=0)
    prec     = precision_score(y_true, y_pred, zero_division=0)
    rec      = recall_score(y_true, y_pred, zero_division=0)
    leakage  = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # FN / positives
    overkill = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # FP / negatives

    print("\n" + "=" * 55)
    print("UNIVID Evaluation Results")
    print("=" * 55)
    print(f"  Violation F1        : {f1:.4f}")
    print(f"  Precision           : {prec:.4f}")
    print(f"  Recall              : {rec:.4f}")
    print(f"  Leakage Rate  (FN%) : {leakage:.4f}  (↓ lower is better)")
    print(f"  Overkill Rate (FP%) : {overkill:.4f}  (↓ lower is better)")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:4d}  FP={fp:4d}")
    print(f"  FN={fn:4d}  TP={tp:4d}")

    # ── Per-policy breakdown ──
    print("\nPer-Policy Breakdown:")
    print(f"  {'Policy':<30} {'F1':>6} {'Leak%':>7} {'Overkill%':>10}")
    print("  " + "-" * 58)
    for pid, pname in enumerate(POLICIES):
        idxs = [i for i, p in enumerate(all_policy_ids) if p == pid]
        if not idxs:
            continue
        yp = [y_pred[i] for i in idxs]
        yt = [y_true[i] for i in idxs]
        pf1 = f1_score(yt, yp, zero_division=0)
        pcm = confusion_matrix(yt, yp, labels=[0, 1])
        if pcm.shape == (2, 2):
            ptn, pfp, pfn, ptp = pcm.ravel()
            pleak  = pfn / (pfn + ptp) if (pfn + ptp) > 0 else 0.0
            pover  = pfp / (pfp + ptn) if (pfp + ptn) > 0 else 0.0
        else:
            pleak, pover = 0.0, 0.0
        print(f"  {pname:<30} {pf1:>6.3f} {pleak*100:>6.1f}%  {pover*100:>9.1f}%")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/univid.pt")
    parser.add_argument("--data", type=str, default="data/univid_toy.pt")
    args = parser.parse_args()
    evaluate(args)
