#!/usr/bin/env python
"""Evaluate dual-path hybrid moderation framework."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.dataset import get_loaders
from src.models import ClassificationPath, SimilarityPath, HybridModerator
from src.retrieval import build_reference_bank

def evaluate_path(scores, labels, threshold=0.5):
    preds = (scores > threshold).float()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return precision, recall

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_loader = get_loaders(args.data_dir)

    # Load trained models
    cls_path = ClassificationPath(384, 256, 5).to(device)
    cls_path.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "cls_path.pt"),
                                         map_location=device))

    sim_path = SimilarityPath(modal_dim=128, embed_dim=64).to(device)
    sim_path.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "sim_path.pt"),
                                         map_location=device))

    # Build reference bank from val set violations
    ref_bank = build_reference_bank(sim_path, val_loader, device=device)
    ref_bank_tensor = ref_bank.build_index().to(device)

    cls_scores, sim_scores, hybrid_scores, all_labels = [], [], [], []

    cls_path.eval(); sim_path.eval()
    with torch.no_grad():
        for batch in val_loader:
            fused = batch["fused"].to(device)
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            is_viol = batch["is_violation"]

            cls_logits = cls_path(fused)
            cls_prob = torch.softmax(cls_logits, dim=-1)[:, 1:].sum(1)
            sim_score = sim_path(text, audio, visual, ref_bank_tensor)
            hyb = 0.5 * cls_prob + 0.5 * sim_score

            cls_scores.append(cls_prob.cpu())
            sim_scores.append(sim_score.cpu())
            hybrid_scores.append(hyb.cpu())
            all_labels.append(is_viol)

    cls_scores = torch.cat(cls_scores)
    sim_scores = torch.cat(sim_scores)
    hybrid_scores = torch.cat(hybrid_scores)
    all_labels = torch.cat(all_labels)

    # Find threshold for ~80% precision per path
    for path_name, scores in [("Classification", cls_scores),
                               ("Similarity", sim_scores),
                               ("Hybrid", hybrid_scores)]:
        best_thr = 0.5
        for thr in torch.linspace(0.1, 0.95, 100):
            p, r = evaluate_path(scores, all_labels, thr.item())
            if p >= 0.78:  # ~80% precision target
                best_thr = thr.item()
                break
        p, r = evaluate_path(scores, all_labels, best_thr)
        print(f"{path_name:15s} @ thr={best_thr:.2f}: Precision={p:.3f}, Recall={r:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/toy_cases")
    parser.add_argument("--ckpt_dir", default="data/ckpt")
    main(parser.parse_args())
