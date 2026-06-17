"""
Training script for UNIVID components.

Paper (§3.4): Three training phases:
  Phase 1 – Risk Filter: BCE with asymmetric FN weighting
  Phase 2 – Moderation Actor (Lite + RAG): CE + caption alignment
  Phase 3 – Trend Governance: online fine-tuning on emerging clusters

Each phase uses synthetic toy data; production training uses expert-labeled
datasets with policy-aware caption annotations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
import numpy as np

from data.toy_dataset import ToyVideoDataset, get_dataloaders
from models.risk_filter import (
    RiskFilter, TextFeatureExtractor, RiskFilterLoss, simple_tokenize
)
from models.policy_caption import (
    PolicyCaptionEncoder, PolicyCaptionDecoder, PolicyAlignedEmbedding, PolicyCaptionLoss
)
from models.univid_lite import UniVIDLite, UniVIDLiteLoss
from models.univid_rag import UniVIDRAG, CaseLibrary, InfoNCELoss


# ── Phase 1: Risk Filter ─────────────────────────────────────────────────────

def train_risk_filter(num_epochs: int = 5, device: str = "cpu"):
    print("=== Phase 1: Training Risk Filter ===")
    train_loader, val_loader, _ = get_dataloaders(batch_size=64, num_samples=1000)

    text_ext = TextFeatureExtractor(vocab_size=10000, embed_dim=128, out_dim=256).to(device)
    filter_model = RiskFilter(visual_dim=768, text_dim=256, hidden_dim=256).to(device)
    criterion = RiskFilterLoss(fn_weight=3.0)
    optimizer = optim.AdamW(
        list(text_ext.parameters()) + list(filter_model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )

    best_recall = 0.0
    for epoch in range(num_epochs):
        filter_model.train(); text_ext.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Filter epoch {epoch+1}", leave=False):
            visual_feat = batch["visual_feat"].to(device)
            labels = batch["label"].to(device)
            token_ids = simple_tokenize(batch["text"]).to(device)

            text_feat = text_ext(token_ids)
            risk_scores = filter_model(visual_feat, text_feat)
            loss = criterion(risk_scores, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(filter_model.parameters()) + list(text_ext.parameters()), 1.0
            )
            optimizer.step()
            total_loss += loss.item()

        # Validate: compute recall at 80% precision
        filter_model.eval(); text_ext.eval()
        scores, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                visual_feat = batch["visual_feat"].to(device)
                token_ids = simple_tokenize(batch["text"]).to(device)
                text_feat = text_ext(token_ids)
                scores.extend(filter_model(visual_feat, text_feat).cpu().tolist())
                gts.extend(batch["label"].tolist())

        precision, recall, _ = precision_recall_curve(gts, scores)
        valid = np.where(precision >= 0.80)[0]
        recall_80p = float(recall[valid[np.argmax(recall[valid])]]) if len(valid) > 0 else 0.0
        best_recall = max(best_recall, recall_80p)
        print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f} | Recall@80P={recall_80p:.3f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "filter_state": filter_model.state_dict(),
        "text_ext_state": text_ext.state_dict(),
    }, "checkpoints/risk_filter.pt")
    print(f"Risk Filter saved. Best Recall@80P: {best_recall:.3f}\n")
    return filter_model, text_ext


# ── Phase 2: Moderation Actor (UNIVID-Lite) ──────────────────────────────────

def train_univid_lite(num_epochs: int = 8, device: str = "cpu"):
    print("=== Phase 2: Training UNIVID-Lite (Moderation Actor) ===")
    train_loader, val_loader, _ = get_dataloaders(batch_size=32, num_samples=1000)

    text_ext = TextFeatureExtractor(vocab_size=10000, embed_dim=128, out_dim=256).to(device)
    cap_encoder = PolicyCaptionEncoder(visual_dim=768, text_dim=256, latent_dim=512).to(device)
    cap_decoder = PolicyCaptionDecoder(latent_dim=512, caption_dim=512).to(device)
    lite = UniVIDLite(visual_dim=768, caption_dim=512, hidden_dim=512,
                      num_categories=8, nhead=8, num_layers=3).to(device)

    criterion_lite = UniVIDLiteLoss(lambda_cat=0.5)
    optimizer = optim.AdamW(
        list(text_ext.parameters()) + list(cap_encoder.parameters()) +
        list(cap_decoder.parameters()) + list(lite.parameters()),
        lr=5e-4, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    for epoch in range(num_epochs):
        lite.train(); cap_encoder.train(); cap_decoder.train(); text_ext.train()
        total_loss = 0.0
        correct = 0; total = 0

        for batch in tqdm(train_loader, desc=f"Lite epoch {epoch+1}", leave=False):
            visual_feat = batch["visual_feat"].to(device)
            labels = batch["label"].to(device)
            # Map label=1 (violation) to category index; 0 (safe) → category 0
            cat_labels = torch.where(
                labels == 0,
                torch.zeros_like(labels),
                torch.clamp(torch.randint_like(labels, 1, 8), 1, 7)
            )

            token_ids = simple_tokenize(batch["text"]).to(device)
            text_feat = text_ext(token_ids)
            latent = cap_encoder(visual_feat, text_feat)
            cap_emb = cap_decoder(latent)
            binary_logits, cat_logits, _ = lite(visual_feat, cap_emb)

            loss = criterion_lite(binary_logits, cat_logits, labels, cat_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(lite.parameters()) + list(cap_encoder.parameters()) +
                list(cap_decoder.parameters()) + list(text_ext.parameters()), 1.0
            )
            optimizer.step()

            total_loss += loss.item()
            preds = binary_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
        acc = correct / total
        best_acc = max(best_acc, acc)
        print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f} | Acc={acc:.3f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "text_ext_state": text_ext.state_dict(),
        "cap_encoder_state": cap_encoder.state_dict(),
        "cap_decoder_state": cap_decoder.state_dict(),
        "lite_state": lite.state_dict(),
    }, "checkpoints/univid_lite.pt")
    print(f"UNIVID-Lite saved. Best Acc: {best_acc:.3f}\n")
    return text_ext, cap_encoder, cap_decoder, lite


# ── Phase 2b: Build Case Library for UNIVID-RAG ──────────────────────────────

def build_case_library(
    text_ext, cap_encoder, cap_decoder, univid_rag, device: str = "cpu"
):
    print("=== Building Case Library for UNIVID-RAG ===")
    train_ds = ToyVideoDataset(num_samples=800, seed=42, violation_rate=0.35)
    case_lib = CaseLibrary(embed_dim=512)

    text_ext.eval(); cap_encoder.eval(); cap_decoder.eval(); univid_rag.eval()
    with torch.no_grad():
        for seg in train_ds.samples:
            if seg.label == 1:
                visual_feat = seg.visual_feat.unsqueeze(0).to(device)
                token_ids = simple_tokenize([seg.text]).to(device)
                text_feat = text_ext(token_ids)
                latent = cap_encoder(visual_feat, text_feat)
                cap_emb = cap_decoder(latent)
                query_emb = univid_rag.query_encoder(visual_feat, cap_emb).squeeze(0)
                case_lib.add(
                    case_id=seg.segment_id,
                    category=seg.category,
                    embed=query_emb.cpu(),
                )

    case_lib.build()
    print(f"Case library built: {len(case_lib)} entries")
    return case_lib


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}\n")

    # Phase 1
    filter_model, text_ext_filter = train_risk_filter(num_epochs=5, device=device)

    # Phase 2
    text_ext, cap_encoder, cap_decoder, lite = train_univid_lite(num_epochs=8, device=device)

    # Phase 2b: Case Library
    rag = UniVIDRAG(visual_dim=768, caption_dim=512, embed_dim=512,
                    num_categories=8, top_k=5).to(device)
    case_lib = build_case_library(text_ext, cap_encoder, cap_decoder, rag, device=device)

    torch.save({
        "rag_state": rag.state_dict(),
        "case_library": case_lib,
    }, "checkpoints/univid_rag.pt")
    print("UNIVID-RAG case library saved.")

    print("\nDone. Checkpoints saved to checkpoints/")
    print("Paper results:")
    print("  Risk Filter:   > 95% recall at 60% precision (pre-screen)")
    print("  UNIVID-Lite:   Standard violation categories")
    print("  UNIVID-RAG:    +9% recall on novel categories vs Lite-only")
    print("  Trend system:  Detects new patterns within hours of emergence")


if __name__ == "__main__":
    main()
