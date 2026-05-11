"""
ARGUS — Evaluation script
Evaluates ad governance model with PDU-augmented analysis.

Usage:
    python evaluate.py --checkpoint checkpoints/argus_stage3_v2_edu_anxiety.pt
                       --policy_version v2_edu_anxiety
"""

import argparse
import logging
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)

from data import AdGovernanceDataset, PolicyKnowledgeBase
from model import PolicyVLM, PDUArchitecture

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate(
    model: PolicyVLM,
    dataset: AdGovernanceDataset,
    policy_kb: PolicyKnowledgeBase,
    use_pdu: bool = False,
    device: str = "cpu",
) -> dict:
    """
    Evaluate model on ad governance dataset.

    Args:
        use_pdu: If True, use PDU umpire verdict as final prediction
                 (Stage II evaluation mode).
    """
    model.eval().to(device)
    pdu = PDUArchitecture(model, policy_kb) if use_pdu else None

    all_preds, all_labels, all_gray = [], [], []

    with torch.no_grad():
        for sample in dataset:
            image = sample["image"].to(device)
            text = sample["text"]
            label = sample["label"].item()

            if use_pdu:
                pdu_result = pdu.rectify(image, text)
                pred = pdu_result.umpire_verdict
            else:
                preds, _ = model.classify(image.unsqueeze(0), [text])
                pred = preds[0].item()

            all_preds.append(pred)
            all_labels.append(label)
            all_gray.append(sample["is_gray_area"])

    # Overall metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="binary", zero_division=0)

    # Gray-area metrics (Historical Recall — key metric from paper)
    gray_indices = [i for i, g in enumerate(all_gray) if g]
    gray_preds = [all_preds[i] for i in gray_indices]
    gray_labels = [all_labels[i] for i in gray_indices]
    gray_recall = recall_score(gray_labels, gray_preds, average="binary", zero_division=0) \
        if gray_indices else 0.0

    results = {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "historical_recall_gray_area": gray_recall,
        "num_gray_area": len(gray_indices),
        "total": len(all_labels),
    }
    return results


def parse_args():
    p = argparse.ArgumentParser(description="ARGUS evaluation")
    p.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    p.add_argument("--policy_version", default="v2_edu_anxiety")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--use_pdu", action="store_true",
                   help="Use PDU umpire verdict (Stage II mode)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()

    dataset = AdGovernanceDataset(
        policy_version=args.policy_version,
        split="test",
        num_samples=args.num_samples,
        seed=999,
    )
    policy_kb = PolicyKnowledgeBase(policy_version=args.policy_version)
    model = PolicyVLM(device=args.device)

    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(state)

    logger.info(f"Evaluating | Policy={args.policy_version} | PDU={args.use_pdu}")

    results = evaluate(model, dataset, policy_kb, use_pdu=args.use_pdu, device=args.device)

    print("\n" + "="*60)
    print(f"  ARGUS Evaluation Results  (policy={args.policy_version})")
    print("="*60)
    print(f"  Accuracy:                    {results['accuracy']:.4f}")
    print(f"  F1 Score:                    {results['f1']:.4f}")
    print(f"  Precision:                   {results['precision']:.4f}")
    print(f"  Recall:                      {results['recall']:.4f}")
    print(f"  Historical Recall (gray):    {results['historical_recall_gray_area']:.4f}  ← key paper metric")
    print(f"  Gray-area samples:           {results['num_gray_area']} / {results['total']}")
    print("="*60)


if __name__ == "__main__":
    main()
