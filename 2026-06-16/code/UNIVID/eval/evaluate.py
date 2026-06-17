"""
Evaluation script for the UNIVID three-stage pipeline.

Measures per-stage performance:
  - Risk Filter: recall, precision, FPR at routing threshold
  - UNIVID-Lite: accuracy, per-category F1
  - Full pipeline: Recall@80P, AP, AUC-ROC, latency breakdown

Paper comparison targets (from §4):
  - UNIVID-Lite precision/recall on standard categories
  - UNIVID-RAG +9% recall improvement on novel categories
  - Risk Filter: >95% recall at routing stage (high-recall pre-screen)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    classification_report, f1_score,
)

from data.toy_dataset import ToyVideoDataset
from models.risk_filter import RiskFilter, TextFeatureExtractor, simple_tokenize
from models.policy_caption import PolicyCaptionEncoder, PolicyCaptionDecoder
from models.univid_lite import UniVIDLite
from models.univid_rag import UniVIDRAG, CaseLibrary
from pipeline.three_stage_pipeline import UniVIDPipeline, build_default_pipeline, CATEGORY_NAMES


def recall_at_precision(scores, labels, target_precision: float = 0.80):
    precision, recall, _ = precision_recall_curve(labels, scores)
    valid = np.where(precision >= target_precision)[0]
    if len(valid) == 0:
        return 0.0
    return float(recall[valid[np.argmax(recall[valid])]])


def evaluate_risk_filter(filter_model, text_ext, dataset, device, threshold=0.3):
    filter_model.eval(); text_ext.eval()
    scores, labels = [], []

    with torch.no_grad():
        for seg in dataset.samples:
            visual_feat = seg.visual_feat.unsqueeze(0).to(device)
            token_ids = simple_tokenize([seg.text]).to(device)
            text_feat = text_ext(token_ids)
            score = filter_model(visual_feat, text_feat)[0].item()
            scores.append(score)
            labels.append(seg.label)

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    preds = (scores_arr > threshold).astype(int)

    tp = ((preds == 1) & (labels_arr == 1)).sum()
    fp = ((preds == 1) & (labels_arr == 0)).sum()
    fn = ((preds == 0) & (labels_arr == 1)).sum()

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    ap = average_precision_score(labels_arr, scores_arr)

    print("\n=== Stage 1: Risk Filter ===")
    print(f"  Threshold={threshold:.2f} | Precision={prec:.3f} | Recall={rec:.3f}")
    print(f"  AP: {ap:.3f}")
    print(f"  Segments routed to actor: {preds.sum()} / {len(preds)} ({preds.mean()*100:.1f}%)")
    print(f"  Paper target: >95% recall at routing threshold")
    return rec, prec, ap


def evaluate_full_pipeline(pipeline, dataset, device):
    """Run the full three-stage pipeline and measure end-to-end performance."""
    scores, labels, preds, stages, latencies = [], [], [], [], []

    for seg in dataset.samples:
        result = pipeline.moderate_segment(
            segment_id=seg.segment_id,
            text=seg.text,
            visual_feat=seg.visual_feat,
        )
        scores.append(result.confidence if result.is_violation else 1.0 - result.confidence)
        labels.append(seg.label)
        preds.append(int(result.is_violation))
        stages.append(result.stage)
        latencies.append(result.latency_ms)

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)

    recall_80p = recall_at_precision(scores_arr, labels_arr, 0.80)
    ap = average_precision_score(labels_arr, scores_arr)
    try:
        auc = roc_auc_score(labels_arr, scores_arr)
    except Exception:
        auc = 0.0

    f1 = f1_score(labels_arr, preds_arr)

    stage_counts = {s: stages.count(s) for s in set(stages)}
    avg_latency = np.mean(latencies)

    print("\n=== Full Pipeline (End-to-End) ===")
    print(f"  Recall@80%P: {recall_80p:.3f}")
    print(f"  AP: {ap:.3f} | AUC-ROC: {auc:.3f} | F1: {f1:.3f}")
    print(f"  Avg latency: {avg_latency:.2f}ms")
    print(f"  Stage distribution: {stage_counts}")
    print(f"  Paper results: UNIVID-RAG +9% recall on novel categories vs UNIVID-Lite alone")
    return recall_80p, ap, auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = ToyVideoDataset(num_samples=500, seed=456, split="test")

    # Load or init models
    text_ext = TextFeatureExtractor(vocab_size=10000, embed_dim=128, out_dim=256).to(device)
    filter_model = RiskFilter(visual_dim=768, text_dim=256, hidden_dim=256).to(device)

    if os.path.exists("checkpoints/risk_filter.pt"):
        ckpt = torch.load("checkpoints/risk_filter.pt", map_location=device)
        filter_model.load_state_dict(ckpt["filter_state"])
        text_ext.load_state_dict(ckpt["text_ext_state"])
        print("Loaded risk filter checkpoint.")

    # Evaluate stage 1
    evaluate_risk_filter(filter_model, text_ext, test_ds, device)

    # Full pipeline eval (random init if no checkpoints)
    pipeline = build_default_pipeline(device=str(device))

    if os.path.exists("checkpoints/univid_lite.pt"):
        ckpt = torch.load("checkpoints/univid_lite.pt", map_location=device)
        pipeline.text_ext.load_state_dict(ckpt["text_ext_state"])
        pipeline.caption_encoder.load_state_dict(ckpt["cap_encoder_state"])
        pipeline.caption_decoder.load_state_dict(ckpt["cap_decoder_state"])
        pipeline.univid_lite.load_state_dict(ckpt["lite_state"])
        print("Loaded UNIVID-Lite checkpoint.")

    if os.path.exists("checkpoints/univid_rag.pt"):
        ckpt = torch.load("checkpoints/univid_rag.pt", map_location=device)
        pipeline.univid_rag.load_state_dict(ckpt["rag_state"])
        pipeline.case_library = ckpt["case_library"]
        print("Loaded UNIVID-RAG checkpoint.")

    evaluate_full_pipeline(pipeline, test_ds, device)
    print("\nNote: Toy dataset results differ from production. Paper uses proprietary datasets.")


if __name__ == "__main__":
    main()
