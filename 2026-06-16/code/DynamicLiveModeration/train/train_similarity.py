"""
Training script for the similarity matching pipeline.

Paper: §3.2 + §3.3
- Trains SimilarityStudent via contrastive distillation from MLLM teacher
- Builds reference bank from training violations
- Calibrates decision threshold to target precision
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from tqdm import tqdm

from data.toy_dataset import ToyLivestreamDataset, get_dataloaders
from models.mllm_teacher import MLLMTeacher, TextEncoder, simple_tokenize
from models.similarity_pipeline import (
    SimilarityStudent, SimilarityDistillationLoss, ReferenceBank
)


def train_epoch(student, teacher, text_enc, optimizer, dataloader, criterion, device):
    student.train()
    teacher.eval()
    text_enc.eval()

    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training similarity", leave=False):
        audio_feat = batch["audio_feat"].to(device)
        visual_feat = batch["visual_feat"].to(device)
        token_ids = simple_tokenize(batch["text"]).to(device)

        with torch.no_grad():
            text_feat = text_enc(token_ids)
            _, teacher_embeds = teacher(text_feat, audio_feat, visual_feat)

        optimizer.zero_grad()
        student_embeds = student(text_feat, audio_feat, visual_feat)
        loss = criterion(student_embeds, teacher_embeds)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def build_reference_bank(student, text_enc, dataset, device):
    """
    Build reference bank from violation samples in training set.
    Paper (§3.2): reference bank = curated known violation examples.
    """
    student.eval()
    text_enc.eval()
    bank = ReferenceBank(embed_dim=512)

    for seg in dataset.samples:
        if seg.label == 1:
            with torch.no_grad():
                text_feat = text_enc(simple_tokenize([seg.text]).to(device))
                audio_feat = seg.audio_feat.unsqueeze(0).to(device)
                visual_feat = seg.visual_feat.unsqueeze(0).to(device)
                embed = student(text_feat, audio_feat, visual_feat).squeeze(0)
            bank.add(seg.category, embed.cpu())

    bank.build()
    print(f"Reference bank built: {sum(len(v) for v in bank.bank.values())} entries")
    return bank


def calibrate_threshold(student, text_enc, bank, val_dataset, device, target_precision=0.80):
    """
    Calibrate similarity threshold to achieve target precision.
    Paper: "calibrated to achieve 80% precision."
    """
    from sklearn.metrics import precision_recall_curve
    import numpy as np

    student.eval()
    text_enc.eval()
    scores, labels = [], []

    all_refs = bank.get_all_refs().to(device)

    for seg in val_dataset.samples:
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

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    valid = np.where(precision >= target_precision)[0]
    if len(valid) == 0:
        best_threshold = 0.5
        best_recall = 0.0
    else:
        best_idx = valid[np.argmax(recall[valid])]
        best_threshold = thresholds[best_idx]
        best_recall = recall[best_idx]

    print(f"Calibrated threshold: {best_threshold:.4f} | "
          f"Recall@{target_precision*100:.0f}%P = {best_recall:.3f}")
    print(f"Paper target: 76% Recall @ 80% Precision")
    return best_threshold, best_recall


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_ds = ToyLivestreamDataset(num_samples=800, seed=42)
    val_ds = ToyLivestreamDataset(num_samples=200, seed=123)
    train_loader, val_loader, _ = get_dataloaders(batch_size=32, num_samples=800)

    teacher = MLLMTeacher(text_dim=128, audio_dim=512, visual_dim=768,
                          hidden_dim=1024, num_classes=2, num_layers=4).to(device)
    text_enc = TextEncoder(vocab_size=1000, embed_dim=64, out_dim=128).to(device)
    student = SimilarityStudent(audio_dim=512, visual_dim=768, text_dim=128,
                                hidden_dim=256, embed_dim=512).to(device)

    print(f"Similarity student params: {sum(p.numel() for p in student.parameters()):,}")

    optimizer = optim.AdamW(
        list(student.parameters()) + list(text_enc.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    criterion = SimilarityDistillationLoss(temperature=0.07)

    for epoch in range(8):
        loss = train_epoch(student, teacher, text_enc, optimizer, train_loader, criterion, device)
        print(f"Epoch {epoch+1:02d} | Distillation Loss={loss:.4f}")

    # Build reference bank
    bank = build_reference_bank(student, text_enc, train_ds, device)

    # Calibrate threshold on validation set
    os.makedirs("checkpoints", exist_ok=True)
    threshold, recall = calibrate_threshold(student, text_enc, bank, val_ds, device)

    torch.save({
        "student_state": student.state_dict(),
        "bank": bank,
        "threshold": threshold,
    }, "checkpoints/similarity_best.pt")
    print(f"Saved. Threshold={threshold:.4f}, Best Recall@80P={recall:.3f}")


if __name__ == "__main__":
    main()
