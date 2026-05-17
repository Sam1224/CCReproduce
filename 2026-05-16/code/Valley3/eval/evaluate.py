"""
Valley3 Evaluation Script
Evaluates the model on e-commerce benchmark tasks.

Benchmark tasks (Valley3 §4):
  1. Product Understanding: attribute prediction, category classification
  2. Livestream Analysis: host/product/audience understanding QA
  3. Moderation & Governance: violation detection F1, false positive rate
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.valley3 import Valley3Model, Valley3Config
from data.toy_ecom_dataset import ToyEcomDataset, VIOLATION_TYPES, collate_fn


def evaluate_governance(model: Valley3Model, loader: DataLoader, device: str) -> dict:
    """
    Evaluate Moderation & Governance task.
    Metrics: Accuracy, Precision, Recall, F1 per violation type.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("violation_label", torch.full((input_ids.size(0),), -1))
            pixel_values = batch.get("pixel_values")
            mel_features = batch.get("mel_features")

            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            if mel_features is not None:
                mel_features = mel_features.to(device)

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                mel_features=mel_features,
            )

            # Use argmax of logits over vocab as proxy classification (toy)
            logits = outputs["logits"][:, 0, :len(VIOLATION_TYPES)]
            preds = logits.argmax(dim=-1).cpu()

            valid_mask = labels >= 0
            if valid_mask.sum() > 0:
                all_preds.extend(preds[valid_mask].tolist())
                all_labels.extend(labels[valid_mask].tolist())

    if not all_labels:
        return {"accuracy": 0.0, "note": "no governance labels in batch"}

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels)

    # Per-class metrics
    class_metrics = {}
    for cls_id, cls_name in enumerate(VIOLATION_TYPES):
        tp = sum(p == cls_id and l == cls_id for p, l in zip(all_preds, all_labels))
        fp = sum(p == cls_id and l != cls_id for p, l in zip(all_preds, all_labels))
        fn = sum(p != cls_id and l == cls_id for p, l in zip(all_preds, all_labels))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        class_metrics[cls_name] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = sum(v["f1"] for v in class_metrics.values()) / len(class_metrics)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": class_metrics,
        "num_samples": len(all_labels),
    }


def evaluate_product_understanding(model: Valley3Model, loader: DataLoader, device: str) -> dict:
    """
    Evaluate Product Understanding task.
    Toy metric: perplexity of ground-truth responses.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch.get("pixel_values")

            if pixel_values is not None:
                pixel_values = pixel_values.to(device)

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=labels,
            )
            if outputs["loss"] is not None:
                valid_tokens = (labels != -100).sum().item()
                total_loss += outputs["loss"].item() * valid_tokens
                total_tokens += valid_tokens

    perplexity = torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()
    return {"perplexity": perplexity, "total_tokens": total_tokens}


def run_evaluation(checkpoint_path: str = None, device: str = "cpu"):
    cfg = Valley3Config(llm_hidden_size=512, num_heads=8, num_layers=2)
    model = Valley3Model(cfg).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint provided — evaluating random initialization (toy demo)")

    # Governance evaluation
    gov_dataset = ToyEcomDataset(stage=3, num_samples=32, seq_len=128)
    gov_loader = DataLoader(gov_dataset, batch_size=8, collate_fn=collate_fn)
    gov_metrics = evaluate_governance(model, gov_loader, device)

    # Product understanding evaluation
    prod_dataset = ToyEcomDataset(stage=2, num_samples=32, seq_len=128)
    prod_loader = DataLoader(prod_dataset, batch_size=8, collate_fn=collate_fn)
    prod_metrics = evaluate_product_understanding(model, prod_loader, device)

    print("\n=== Evaluation Results ===")
    print("\n[Moderation & Governance]")
    print(f"  Accuracy: {gov_metrics.get('accuracy', 0):.4f}")
    print(f"  Macro F1: {gov_metrics.get('macro_f1', 0):.4f}")
    if "per_class" in gov_metrics:
        for cls, m in gov_metrics["per_class"].items():
            print(f"  {cls:30s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    print("\n[Product Understanding]")
    print(f"  Perplexity: {prod_metrics['perplexity']:.2f}")

    return {"governance": gov_metrics, "product_understanding": prod_metrics}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.device)
