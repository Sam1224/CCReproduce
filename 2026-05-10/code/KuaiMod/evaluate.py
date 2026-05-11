"""
KuaiMod Evaluation Script

Evaluates on SVP-style benchmark. Reports:
  - Overall accuracy
  - Per-class precision / recall / F1
  - Macro-F1
  - Sample CoT rationale outputs

Metrics align with paper's evaluation protocol:
  - Primary: Macro-F1 on 3-class classification
  - Secondary: Recall on 'violating' class (recall@P threshold)
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from data import SVPDataset, SVPCollator, ID2LABEL, LABEL2ID
from model import KuaiMod
from train import simple_tokenize, simple_tokenize_cot


@torch.no_grad()
def evaluate(model, loader, device, vocab_size=1000, show_samples=3):
    model.eval()
    all_preds = []
    all_labels = []
    sample_outputs = []

    for batch_idx, batch in enumerate(loader):
        frames = batch["frames"].to(device)
        texts = batch["texts"]
        labels = batch["labels"]

        text_ids = simple_tokenize(texts, vocab_size).to(device)

        pred = model.predict(frames, text_ids)
        verdicts = pred["verdicts"].cpu()
        probs = pred["verdict_probs"].cpu()

        all_preds.extend(verdicts.tolist())
        all_labels.extend(labels.tolist())

        if len(sample_outputs) < show_samples:
            for i in range(min(show_samples - len(sample_outputs), len(texts))):
                sample_outputs.append({
                    "text": texts[i][:80],
                    "true_label": ID2LABEL[labels[i].item()],
                    "pred_label": ID2LABEL[verdicts[i].item()],
                    "probs": {ID2LABEL[j]: f"{probs[i][j]:.3f}" for j in range(3)},
                })

    # Print sample predictions
    print("\n=== Sample Predictions ===")
    for s in sample_outputs:
        match = "✓" if s["true_label"] == s["pred_label"] else "✗"
        print(f"  [{match}] true={s['true_label']:<12} pred={s['pred_label']:<12} "
              f"probs={s['probs']}")
        print(f"      text: {s['text']}...")

    # Metrics
    target_names = [ID2LABEL[i] for i in range(3)]
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=target_names,
                                zero_division=0))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    print(f"  (rows=true, cols=pred)  labels: {target_names}")
    print(cm)

    # Recall@Precision for 'violating' class (key metric in paper)
    tp_v = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp_v = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l != 1)
    fn_v = sum(1 for p, l in zip(all_preds, all_labels) if p != 1 and l == 1)
    prec_v = tp_v / max(tp_v + fp_v, 1)
    rec_v = tp_v / max(tp_v + fn_v, 1)
    print(f"\n=== Violating Class Stats ===")
    print(f"  Precision: {prec_v:.3f}  Recall: {rec_v:.3f}")
    print(f"  (Paper target: ~67% Recall at 80% Precision in production)")

    return {
        "all_preds": all_preds,
        "all_labels": all_labels,
        "violating_precision": prec_v,
        "violating_recall": rec_v,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SVPDataset(num_samples=args.num_samples, split="test", seed=args.seed)
    collator = SVPCollator()
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collator)

    model = KuaiMod(vocab_size=args.vocab_size).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded: {args.checkpoint}")
    else:
        print("No checkpoint provided — using random weights (expect ~33% accuracy)")

    results = evaluate(model, loader, device, vocab_size=args.vocab_size)
    return results


if __name__ == "__main__":
    main()
