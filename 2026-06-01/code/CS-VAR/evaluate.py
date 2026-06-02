"""
Evaluation script for CS-VAR.
Reports risk classification accuracy, AUC, and interpretable localized signals.
"""
import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from model import SessionRiskModel
from retrieval import SessionDatabase
from dataset import get_dataloaders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--db", type=str, default="checkpoints/db.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--k_retrieved", type=int, default=5)
    p.add_argument("--num_risk_levels", type=int, default=3)
    return p.parse_args()


@torch.no_grad()
def run_evaluation(model, loader, db, k, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        events = batch["events"].to(device)
        labels = batch["risk_label"]
        curr_emb = model.encode_session(events)
        retrieved_emb, _ = db.retrieve(curr_emb, k=k)
        retrieved_emb = retrieved_emb.to(device)
        logits, _ = model(events, retrieved_emb)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    return np.array(all_labels), np.array(all_preds), all_probs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, _ = get_dataloaders(args.batch_size)

    model = SessionRiskModel(num_risk_levels=args.num_risk_levels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    db = torch.load(args.db, map_location="cpu")

    y_true, y_pred, y_prob = run_evaluation(model, test_loader, db, args.k_retrieved, device)

    print("=== CS-VAR Evaluation ===\n")
    print(classification_report(y_true, y_pred, target_names=["Low", "Medium", "High"]))

    if args.num_risk_levels == 3:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
            print(f"Macro AUC-ROC: {auc:.4f}")
        except ValueError as e:
            print(f"AUC computation skipped: {e}")

    # High-risk recall (key metric for production)
    high_risk_true = (y_true == 2).astype(int)
    high_risk_pred = (y_pred == 2).astype(int)
    tp = (high_risk_true & high_risk_pred).sum()
    fn = (high_risk_true & ~high_risk_pred.astype(bool)).sum()
    recall_high = tp / (tp + fn + 1e-9)
    print(f"\nHigh-Risk Recall: {recall_high:.4f}")
    print("(Paper: state-of-the-art on large-scale industrial dataset)")


if __name__ == "__main__":
    main()
