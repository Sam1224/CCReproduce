"""
Evaluation script for the dual-pipeline system.

Paper results:
  - Classification pipeline: 67% Recall @ 80% Precision
  - Similarity pipeline: 76% Recall @ 80% Precision
  - A/B test: 6-8% reduction in user views of unwanted livestreams
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    classification_report, roc_auc_score
)

from data.toy_dataset import ToyLivestreamDataset
from models.mllm_teacher import TextEncoder, simple_tokenize
from models.classification_pipeline import ClassificationStudent
from models.similarity_pipeline import SimilarityStudent, ReferenceBank, SimilarityDecider


def recall_at_precision(scores, labels, target_precision=0.80):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    valid = np.where(precision >= target_precision)[0]
    if len(valid) == 0:
        return 0.0
    return float(recall[valid[np.argmax(recall[valid])]])


def evaluate_classification_pipeline(student, text_enc, dataset, device):
    """Evaluate the supervised classification pipeline."""
    student.eval()
    text_enc.eval()
    scores, labels = [], []

    for seg in dataset.samples:
        with torch.no_grad():
            text_feat = text_enc(simple_tokenize([seg.text]).to(device))
            audio_feat = seg.audio_feat.unsqueeze(0).to(device)
            visual_feat = seg.visual_feat.unsqueeze(0).to(device)
            logits = student(text_feat, audio_feat, visual_feat)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()

        scores.append(prob)
        labels.append(seg.label)

    recall_80p = recall_at_precision(scores, labels, 0.80)
    ap = average_precision_score(labels, scores)
    auc = roc_auc_score(labels, scores)

    print("\n=== Classification Pipeline ===")
    print(f"  Recall @ 80% Precision: {recall_80p:.3f}  (Paper: 0.67)")
    print(f"  Average Precision (AP): {ap:.3f}")
    print(f"  AUC-ROC: {auc:.3f}")
    return recall_80p


def evaluate_similarity_pipeline(student, text_enc, bank, dataset, device, threshold=0.5):
    """Evaluate the similarity matching pipeline."""
    student.eval()
    text_enc.eval()
    scores, labels = [], []
    all_refs = bank.get_all_refs().to(device)

    for seg in dataset.samples:
        with torch.no_grad():
            text_feat = text_enc(simple_tokenize([seg.text]).to(device))
            audio_feat = seg.audio_feat.unsqueeze(0).to(device)
            visual_feat = seg.visual_feat.unsqueeze(0).to(device)
            embed = student(text_feat, audio_feat, visual_feat).squeeze(0)

        if len(all_refs) > 0:
            sims = torch.mv(all_refs, embed)
            max_sim = sims.max().item()
        else:
            max_sim = 0.0

        scores.append(max_sim)
        labels.append(seg.label)

    recall_80p = recall_at_precision(scores, labels, 0.80)
    ap = average_precision_score(labels, scores)
    auc = roc_auc_score(labels, scores)

    print("\n=== Similarity Pipeline ===")
    print(f"  Recall @ 80% Precision: {recall_80p:.3f}  (Paper: 0.76)")
    print(f"  Average Precision (AP): {ap:.3f}")
    print(f"  AUC-ROC: {auc:.3f}")
    return recall_80p


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = ToyLivestreamDataset(num_samples=500, seed=456, split="test")

    text_enc = TextEncoder(vocab_size=1000, embed_dim=64, out_dim=128).to(device)
    cls_student = ClassificationStudent(
        audio_dim=512, visual_dim=768, text_dim=128, hidden_dim=256, num_classes=2
    ).to(device)
    sim_student = SimilarityStudent(
        audio_dim=512, visual_dim=768, text_dim=128, hidden_dim=256, embed_dim=512
    ).to(device)

    # Load checkpoints if available
    if os.path.exists("checkpoints/classifier_best.pt"):
        cls_student.load_state_dict(torch.load("checkpoints/classifier_best.pt", map_location=device))
        print("Loaded classifier checkpoint.")

    if os.path.exists("checkpoints/similarity_best.pt"):
        ckpt = torch.load("checkpoints/similarity_best.pt", map_location=device)
        sim_student.load_state_dict(ckpt["student_state"])
        bank = ckpt["bank"]
        threshold = ckpt["threshold"]
        print("Loaded similarity checkpoint.")
    else:
        bank = ReferenceBank(embed_dim=512)
        threshold = 0.5
        print("No similarity checkpoint found, using random init.")

    cls_recall = evaluate_classification_pipeline(cls_student, text_enc, test_ds, device)
    sim_recall = evaluate_similarity_pipeline(sim_student, text_enc, bank, test_ds, device, threshold)

    print(f"\n=== Summary ===")
    print(f"Classification Recall@80P: {cls_recall:.3f} | Paper: 0.67")
    print(f"Similarity    Recall@80P: {sim_recall:.3f} | Paper: 0.76")
    print("Note: Toy dataset results may differ from production metrics.")


if __name__ == "__main__":
    main()
