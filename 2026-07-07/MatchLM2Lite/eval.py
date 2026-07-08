"""
MatchLM2Lite — Evaluation Script

Evaluates both MatchLM (teacher) and MatchLite (student) on the RCI task:
- Binary reproduction detection (threshold on continuous score)
- Metrics: accuracy, precision, recall, F1
- Breakdown by reproduction type

Usage:
    python eval.py --model teacher
    python eval.py --model student
    python eval.py --model both --threshold 0.5
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import RCIDataset, simulate_visual_features, simulate_audio_features
from model import MatchLM, MatchLite
from train import collate_rci


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model,
    dataset: RCIDataset,
    threshold: float = 0.5,
    batch_size: int = 4,
    device: str = "cpu",
    is_student: bool = False,
) -> dict:
    """Compute binary classification metrics at the given threshold."""
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_rci)

    all_scores = []
    all_labels = []
    all_types = []

    # Track per-type results
    scores_by_type: dict[str, list] = {}
    labels_by_type: dict[str, list] = {}

    pair_idx = 0
    for batch in loader:
        B = batch["labels"].size(0)

        if is_student:
            ref_lite, _ = model.encode_video(
                batch["ref_visual"].to(device),
                batch["ref_audio"].to(device),
                batch["ref_input_ids"].to(device),
            )
            cand_lite, _ = model.encode_video(
                batch["cand_visual"].to(device),
                batch["cand_audio"].to(device),
                batch["cand_input_ids"].to(device),
            )
            pair_feats = torch.cat([
                ref_lite, cand_lite,
                ref_lite * cand_lite,
                (ref_lite - cand_lite).abs(),
            ], dim=-1)
            scores = model.scorer(pair_feats).squeeze(-1)
        else:
            output = model(
                ref_visual=batch["ref_visual"].to(device),
                ref_audio=batch["ref_audio"].to(device),
                ref_input_ids=batch["ref_input_ids"].to(device),
                ref_attention_mask=batch["ref_attention_mask"].to(device),
                cand_visual=batch["cand_visual"].to(device),
                cand_audio=batch["cand_audio"].to(device),
                cand_input_ids=batch["cand_input_ids"].to(device),
                cand_attention_mask=batch["cand_attention_mask"].to(device),
            )
            scores = output["reproduction_score"]

        all_scores.extend(scores.cpu().tolist())
        all_labels.extend(batch["labels"].tolist())
        pair_idx += B

    # Collect per-type info from the dataset
    for i, item in enumerate(dataset):
        rtype = item["reproduction_type"] or "no_reproduction"
        if rtype not in scores_by_type:
            scores_by_type[rtype] = []
            labels_by_type[rtype] = []
        if i < len(all_scores):
            scores_by_type[rtype].append(all_scores[i])
            labels_by_type[rtype].append(all_labels[i])

    # Binary predictions at threshold
    preds = [1 if s >= threshold else 0 for s in all_scores]
    gt = [1 if l >= 0.5 else 0 for l in all_labels]

    tp = sum(1 for p, g in zip(preds, gt) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(preds, gt) if p == 1 and g == 0)
    tn = sum(1 for p, g in zip(preds, gt) if p == 0 and g == 0)
    fn = sum(1 for p, g in zip(preds, gt) if p == 0 and g == 1)

    accuracy = (tp + tn) / max(len(preds), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Per-type accuracy
    type_accuracy = {}
    for rtype in scores_by_type:
        s_list = scores_by_type[rtype]
        l_list = labels_by_type[rtype]
        preds_t = [1 if s >= threshold else 0 for s in s_list]
        gt_t = [1 if l >= 0.5 else 0 for l in l_list]
        type_accuracy[rtype] = sum(p == g for p, g in zip(preds_t, gt_t)) / max(len(preds_t), 1)

    return {
        "num_samples": len(preds),
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "per_type_accuracy": type_accuracy,
    }


def print_metrics(name: str, metrics: dict):
    print(f"\n── {name}")
    print(f"   Samples : {metrics['num_samples']}")
    print(f"   Threshold: {metrics['threshold']:.2f}")
    print(f"   Accuracy : {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall   : {metrics['recall']:.4f}")
    print(f"   F1       : {metrics['f1']:.4f}")
    print(f"   TP/FP/TN/FN: {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}")
    if metrics["per_type_accuracy"]:
        print("   Per reproduction type:")
        for rtype, acc in metrics["per_type_accuracy"].items():
            print(f"     {rtype:25s}: {acc:.4f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MatchLM2Lite Evaluation")
    parser.add_argument("--model", choices=["teacher", "student", "both"], default="both")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--embed_dim_teacher", type=int, default=512)
    parser.add_argument("--embed_dim_student", type=int, default=256)
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--student_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RCIDataset(augment=True)
    print(f"Dataset size: {len(dataset)} pairs")

    print("\n" + "=" * 60)
    print("MatchLM2Lite — Reproduced Content Identification Evaluation")
    print("=" * 60)

    if args.model in ("teacher", "both"):
        teacher = MatchLM(embed_dim=args.embed_dim_teacher)
        if args.teacher_ckpt:
            teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location="cpu"))
            print(f"\nLoaded teacher checkpoint from {args.teacher_ckpt}")
        else:
            print("\nNo teacher checkpoint — using random weights (interface demo only)")

        metrics = evaluate_model(teacher, dataset, threshold=args.threshold, device=str(device))
        print_metrics("MatchLM (Teacher)", metrics)

    if args.model in ("student", "both"):
        student = MatchLite(embed_dim=args.embed_dim_student,
                            teacher_embed_dim=args.embed_dim_teacher)
        if args.student_ckpt:
            student.load_state_dict(torch.load(args.student_ckpt, map_location="cpu"))
            print(f"\nLoaded student checkpoint from {args.student_ckpt}")
        else:
            print("\nNo student checkpoint — using random weights (interface demo only)")

        metrics = evaluate_model(student, dataset, threshold=args.threshold,
                                 device=str(device), is_student=True)
        print_metrics("MatchLite (Student)", metrics)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
