"""
UNIVID — Evaluation Script

Evaluates the UNIVID pipeline on a held-out test set.

Metrics computed (matching paper §5.3):
  - Violation Leakage Rate (VLR): FN / (FN + TP)  ← paper reports -42.7% relative
  - Overkill Rate (OKR):          FP / (FP + TN)  ← paper reports -37.0% relative
  - F1, Precision, Recall per violation type
  - Caption quality: BLEU, ROUGE-L (where reference captions available)
"""

import argparse
import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, CLIPImageProcessor
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from model import UNIVID, ModerationClassificationHead
from dataset import build_dataloader, VideoModerationDataset
from pipeline import UNIVIDPipeline


def compute_moderation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, violation_types: list
) -> dict:
    """Compute per-class and macro metrics."""
    results = {}
    for i, vtype in enumerate(violation_types):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", zero_division=0
        )
        results[vtype] = {"precision": p, "recall": r, "f1": f1}

    # Macro
    p, r, f1, _ = precision_recall_fscore_support(
        y_true.ravel(), y_pred.ravel(), average="macro", zero_division=0
    )
    results["macro"] = {"precision": p, "recall": r, "f1": f1}
    return results


def compute_violation_leakage_and_overkill(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """
    Paper's primary metrics:
      Violation Leakage Rate = FN / (TP + FN)  [lower is better]
      Overkill Rate          = FP / (TN + FP)  [lower is better]

    These are measured at the video level (any violation = 1).
    """
    y_true_any = (y_true.sum(axis=1) > 0).astype(int)
    y_pred_any = (y_pred.sum(axis=1) > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_any, y_pred_any, labels=[0, 1]).ravel()
    vlr = fn / (tp + fn + 1e-9)
    okr = fp / (tn + fp + 1e-9)
    return {
        "violation_leakage_rate": vlr,
        "overkill_rate": okr,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

    model = UNIVID(llm_name=args.llm_name).to(device)
    if os.path.exists(os.path.join(args.checkpoint_dir, "univid_final.pt")):
        model.load_state_dict(
            torch.load(
                os.path.join(args.checkpoint_dir, "univid_final.pt"),
                map_location=device,
            )
        )
    model.eval()

    cls_head = ModerationClassificationHead(
        hidden_dim=model.llm.config.hidden_size
    ).to(device)
    if os.path.exists(os.path.join(args.checkpoint_dir, "cls_head.pt")):
        cls_head.load_state_dict(
            torch.load(
                os.path.join(args.checkpoint_dir, "cls_head.pt"),
                map_location=device,
            )
        )
    cls_head.eval()

    test_loader = build_dataloader(
        args.data_path, tokenizer, image_processor, "test", batch_size=1
    )
    violation_types = VideoModerationDataset.VIOLATION_TYPES

    all_true, all_pred = [], []
    threshold = 0.5

    for batch in test_loader:
        pixel_values = batch["pixel_values"].to(device)
        policy_input_ids = batch["policy_input_ids"].to(device)
        violation_labels = batch["violation_labels"].numpy()

        embedding = model.get_caption_embedding(pixel_values, policy_input_ids)
        logits = cls_head(embedding)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)

        all_true.append(violation_labels)
        all_pred.append(preds)

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)

    metrics = compute_moderation_metrics(y_true, y_pred, violation_types)
    vl_metrics = compute_violation_leakage_and_overkill(y_true, y_pred)

    print("\n=== UNIVID Evaluation Results ===")
    print(f"\nViolation Leakage Rate: {vl_metrics['violation_leakage_rate']:.4f}")
    print(f"Overkill Rate:          {vl_metrics['overkill_rate']:.4f}")
    print(f"TP={vl_metrics['tp']} FP={vl_metrics['fp']} "
          f"FN={vl_metrics['fn']} TN={vl_metrics['tn']}")

    print("\n--- Per-class F1 ---")
    for vtype in violation_types:
        m = metrics[vtype]
        print(f"  {vtype:20s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    print(f"\n  {'MACRO':20s}  P={metrics['macro']['precision']:.3f}  "
          f"R={metrics['macro']['recall']:.3f}  F1={metrics['macro']['f1']:.3f}")

    output_path = os.path.join(args.checkpoint_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump({**metrics, **vl_metrics}, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/toy_videos")
    parser.add_argument("--llm_name", default="Qwen/Qwen2-1.5B")
    parser.add_argument("--checkpoint_dir", default="checkpoints/univid")
    args = parser.parse_args()
    evaluate(args)
