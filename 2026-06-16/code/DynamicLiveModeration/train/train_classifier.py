"""
Training script for the supervised classification pipeline.

Paper: §3.1 + §3.3 (with MLLM KD)
- Trains ClassificationStudent with cross-entropy + KD loss
- Teacher: MLLMTeacher (simulated large model)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.toy_dataset import ToyLivestreamDataset, get_dataloaders
from models.mllm_teacher import MLLMTeacher, TextEncoder, simple_tokenize
from models.classification_pipeline import ClassificationStudent, ClassificationLoss


def train_epoch(student, teacher, text_enc, optimizer, dataloader, criterion, device):
    student.train()
    teacher.eval()
    text_enc.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        text_list = batch["text"]
        audio_feat = batch["audio_feat"].to(device)
        visual_feat = batch["visual_feat"].to(device)
        labels = batch["label"].to(device)

        # Encode text
        token_ids = simple_tokenize(text_list).to(device)
        with torch.no_grad():
            text_feat = text_enc(token_ids)

        # Get teacher soft labels
        with torch.no_grad():
            teacher_soft, _ = teacher.get_soft_labels(text_feat, audio_feat, visual_feat)

        # Student forward
        optimizer.zero_grad()
        student_logits = student(text_feat, audio_feat, visual_feat)

        # Combined CE + KD loss
        loss = criterion(student_logits, labels, teacher_soft_labels=teacher_soft)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = student_logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(student, text_enc, dataloader, device):
    student.eval()
    text_enc.eval()

    all_preds, all_labels, all_scores = [], [], []

    for batch in dataloader:
        text_list = batch["text"]
        audio_feat = batch["audio_feat"].to(device)
        visual_feat = batch["visual_feat"].to(device)
        labels = batch["label"].to(device)

        token_ids = simple_tokenize(text_list).to(device)
        text_feat = text_enc(token_ids)
        logits = student(text_feat, audio_feat, visual_feat)
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(violation)

        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_scores.extend(probs.cpu().tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return acc, all_scores, all_labels


def compute_recall_at_precision(scores, labels, target_precision=0.80):
    """Find recall when precision >= target_precision (as in paper)."""
    from sklearn.metrics import precision_recall_curve
    import numpy as np
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # Find highest recall where precision >= target
    valid = np.where(precision >= target_precision)[0]
    if len(valid) == 0:
        return 0.0, 1.0
    best_idx = valid[np.argmax(recall[valid])]
    return recall[best_idx], thresholds[best_idx]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(batch_size=32, num_samples=800)

    # Models
    teacher = MLLMTeacher(text_dim=128, audio_dim=512, visual_dim=768,
                          hidden_dim=1024, num_classes=2, num_layers=4).to(device)
    text_enc = TextEncoder(vocab_size=1000, embed_dim=64, out_dim=128).to(device)
    student = ClassificationStudent(audio_dim=512, visual_dim=768, text_dim=128,
                                     hidden_dim=256, num_classes=2).to(device)

    # Pretend teacher is pretrained (random init here for toy demo)
    print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")

    optimizer = optim.AdamW(
        list(student.parameters()) + list(text_enc.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    criterion = ClassificationLoss(alpha=0.5, temperature=3.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_recall = 0.0
    for epoch in range(10):
        train_loss, train_acc = train_epoch(
            student, teacher, text_enc, optimizer, train_loader, criterion, device
        )
        val_acc, val_scores, val_labels = evaluate(student, text_enc, val_loader, device)
        recall_at_80p, threshold = compute_recall_at_precision(val_scores, val_labels, 0.80)
        scheduler.step()

        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.3f} | "
              f"Val Acc={val_acc:.3f} | "
              f"Recall@80%P={recall_at_80p:.3f} (thresh={threshold:.3f})")

        if recall_at_80p > best_recall:
            best_recall = recall_at_80p
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(student.state_dict(), "checkpoints/classifier_best.pt")

    print(f"\nBest Recall@80%P: {best_recall:.3f}")
    print("Paper target (production): 67% Recall @ 80% Precision")


if __name__ == "__main__":
    main()
