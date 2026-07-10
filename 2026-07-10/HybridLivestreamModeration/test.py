import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, ToyLivestreamModerationDataset
from model import HybridLivestreamModerationModel


def _move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def recall_at_precision(scores: torch.Tensor, targets: torch.Tensor, target_precision: float) -> float:
    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    best_recall = 0.0
    for threshold in sorted_scores:
        preds = scores >= threshold
        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision >= target_precision:
            best_recall = max(best_recall, recall)
    return best_recall


def macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    scores = []
    for class_id in range(num_classes):
        tp = ((preds == class_id) & (targets == class_id)).sum().item()
        fp = ((preds == class_id) & (targets != class_id)).sum().item()
        fn = ((preds != class_id) & (targets == class_id)).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = ToyLivestreamModerationDataset(length=args.samples)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model = HybridLivestreamModerationModel(num_classes=len(CLASS_NAMES)).to(device)
    checkpoint_path = Path(args.checkpoint)
    reference_bank = dataset.build_reference_bank()
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        reference_bank.frames = checkpoint["reference_bank"]["frames"]
        reference_bank.audio = checkpoint["reference_bank"]["audio"]
        reference_bank.text = checkpoint["reference_bank"]["text"]
        reference_bank.labels = checkpoint["reference_bank"]["labels"]

    reference_bank_tensors = {
        "frames": reference_bank.frames.to(device),
        "audio": reference_bank.audio.to(device),
        "text": reference_bank.text.to(device),
    }
    reference_labels = reference_bank.labels.to(device)
    model.eval()

    all_logits = []
    all_labels = []
    risky_scores = []
    risky_targets = []
    retrieval_hits_top1 = []
    retrieval_hits_top5 = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            outputs = model(batch["frames"], batch["audio"], batch["text"], reference_bank_tensors)
            logits = outputs["classifier_logits"]
            probs = torch.softmax(logits, dim=-1)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())
            risky_scores.append((1.0 - probs[:, 0]).cpu())
            risky_targets.append((batch["label"] != 0).long().cpu())

            rerank = outputs["rerank_scores"]
            for row_scores, label in zip(rerank, batch["label"]):
                if label.item() == 0:
                    continue
                top1 = torch.topk(row_scores, k=1).indices
                top5 = torch.topk(row_scores, k=min(5, row_scores.numel())).indices
                retrieval_hits_top1.append(int((reference_labels[top1] == label).any().item()))
                retrieval_hits_top5.append(int((reference_labels[top5] == label).any().item()))

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = logits.argmax(dim=-1)
    risk_scores = torch.cat(risky_scores)
    risk_targets = torch.cat(risky_targets)

    report = {
        "toy_macro_f1": round(macro_f1(preds, labels, len(CLASS_NAMES)), 4),
        "toy_recall_at_p80": round(recall_at_precision(risk_scores, risk_targets, 0.80), 4),
        "toy_retrieval_recall_top1": round(sum(retrieval_hits_top1) / max(1, len(retrieval_hits_top1)), 4),
        "toy_retrieval_recall_top5": round(sum(retrieval_hits_top5) / max(1, len(retrieval_hits_top5)), 4),
    }
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/hybrid_livestream_moderation.pt")
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())
