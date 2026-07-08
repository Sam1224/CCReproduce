"""
MatchLM2Lite — Training Script

Implements:
1. MatchLM teacher training (binary cross-entropy on RCI pairs)
2. MatchLite student training with knowledge distillation
   - Hard label supervision (BCE)
   - Soft embedding distillation (MSE alignment to teacher)

Usage:
    python train.py --mode teacher --epochs 5
    python train.py --mode student --epochs 5 --distill_weight 0.5
    python train.py --mode both --epochs 5

Paper: "MatchLM2Lite: A Scalable MLLM-to-Lite Framework for Reproduced Content Identification"
arXiv: https://arxiv.org/abs/2606.14786
"""

import argparse
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import RCIDataset, simulate_visual_features, simulate_audio_features
from model import MatchLM, MatchLite


# ─── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_rci(batch: list[dict], max_len: int = 64, vocab_size: int = 32000,
                num_frames: int = 8, num_segments: int = 4,
                visual_dim: int = 512, audio_dim: int = 256) -> dict:
    """Collate RCI pairs: simulate visual/audio features + tokenize text."""
    B = len(batch)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float)

    def tokenize(text: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [min(ord(c), vocab_size - 1) for c in text[:max_len]]
        pad = max_len - len(ids)
        return (
            torch.tensor(ids + [0] * pad, dtype=torch.long),
            torch.tensor([1] * len(ids) + [0] * pad, dtype=torch.long),
        )

    ref_texts = [item["ref_title"] + " " + item["ref_description"] + " " + item["ref_captions"]
                 for item in batch]
    cand_texts = [item["cand_title"] + " " + item["cand_description"] + " " + item["cand_captions"]
                  for item in batch]

    ref_ids, ref_masks = zip(*[tokenize(t) for t in ref_texts])
    cand_ids, cand_masks = zip(*[tokenize(t) for t in cand_texts])

    return {
        # Simulated visual features (production: ViT/CLIP features)
        "ref_visual": simulate_visual_features(B, num_frames, visual_dim),
        "ref_audio": simulate_audio_features(B, num_segments, audio_dim),
        "ref_input_ids": torch.stack(ref_ids),
        "ref_attention_mask": torch.stack(ref_masks),
        # Candidate
        "cand_visual": simulate_visual_features(B, num_frames, visual_dim),
        "cand_audio": simulate_audio_features(B, num_segments, audio_dim),
        "cand_input_ids": torch.stack(cand_ids),
        "cand_attention_mask": torch.stack(cand_masks),
        # Labels
        "labels": labels,
    }


# ─── Teacher Training ─────────────────────────────────────────────────────────

def train_teacher(
    model: MatchLM,
    dataset: RCIDataset,
    num_epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cpu",
) -> dict:
    """Train the MatchLM teacher on binary reproduction classification."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_rci)

    history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in loader:
            optimizer.zero_grad()
            output = model(
                ref_visual=batch["ref_visual"].to(device),
                ref_audio=batch["ref_audio"].to(device),
                ref_input_ids=batch["ref_input_ids"].to(device),
                ref_attention_mask=batch["ref_attention_mask"].to(device),
                cand_visual=batch["cand_visual"].to(device),
                cand_audio=batch["cand_audio"].to(device),
                cand_input_ids=batch["cand_input_ids"].to(device),
                cand_attention_mask=batch["cand_attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        history.append(avg_loss)
        print(f"  [Teacher] Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.4f}")

    return {"teacher_history": history}


# ─── Student Distillation Training ────────────────────────────────────────────

@torch.no_grad()
def extract_teacher_embeddings(
    teacher: MatchLM,
    batch: dict,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run teacher forward pass to get reference and candidate embeddings."""
    teacher.eval()
    ref_emb = teacher.encode_video(
        batch["ref_visual"].to(device),
        batch["ref_audio"].to(device),
        batch["ref_input_ids"].to(device),
        batch["ref_attention_mask"].to(device),
    )
    cand_emb = teacher.encode_video(
        batch["cand_visual"].to(device),
        batch["cand_audio"].to(device),
        batch["cand_input_ids"].to(device),
        batch["cand_attention_mask"].to(device),
    )
    return ref_emb, cand_emb


def train_student(
    teacher: MatchLM,
    student: MatchLite,
    dataset: RCIDataset,
    num_epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    distill_weight: float = 0.5,
    device: str = "cpu",
) -> dict:
    """
    Train MatchLite student with knowledge distillation from MatchLM teacher.

    The combined loss:
        L = (1 - distill_weight) * L_label + distill_weight * L_distill
    where:
        L_label = BCE(student_score, binary_label)
        L_distill = 0.5 * [MSE(aligned_ref, teacher_ref) + MSE(aligned_cand, teacher_cand)]
    """
    teacher = teacher.to(device).eval()
    student = student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_rci)

    history = []
    for epoch in range(num_epochs):
        student.train()
        epoch_loss = 0.0
        epoch_label = 0.0
        epoch_distill = 0.0
        num_batches = 0

        for batch in loader:
            optimizer.zero_grad()

            # Get teacher embeddings (no grad)
            teacher_ref_emb, teacher_cand_emb = extract_teacher_embeddings(
                teacher, batch, device
            )

            output = student(
                ref_visual=batch["ref_visual"].to(device),
                ref_audio=batch["ref_audio"].to(device),
                ref_input_ids=batch["ref_input_ids"].to(device),
                cand_visual=batch["cand_visual"].to(device),
                cand_audio=batch["cand_audio"].to(device),
                cand_input_ids=batch["cand_input_ids"].to(device),
                labels=batch["labels"].to(device),
                teacher_ref_emb=teacher_ref_emb,
                teacher_cand_emb=teacher_cand_emb,
                distill_weight=distill_weight,
            )

            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            if "label_loss" in output:
                epoch_label += output["label_loss"]
                epoch_distill += output["distill_loss"]
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        history.append(avg_loss)
        if epoch_label > 0:
            print(f"  [Student] Epoch {epoch+1}/{num_epochs} — "
                  f"total: {avg_loss:.4f}, "
                  f"label: {epoch_label/num_batches:.4f}, "
                  f"distill: {epoch_distill/num_batches:.4f}")
        else:
            print(f"  [Student] Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.4f}")

    return {"student_history": history}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MatchLM2Lite Training")
    parser.add_argument("--mode", choices=["teacher", "student", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--embed_dim_teacher", type=int, default=512)
    parser.add_argument("--embed_dim_student", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--distill_weight", type=float, default=0.5,
                        help="Balance between label loss and embedding distillation loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = RCIDataset(augment=True)
    print(f"Dataset size: {len(dataset)} pairs")

    # Models
    teacher = MatchLM(embed_dim=args.embed_dim_teacher)
    student = MatchLite(embed_dim=args.embed_dim_student,
                        teacher_embed_dim=args.embed_dim_teacher)

    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print(f"MatchLM (teacher) parameters: {t_params:,}")
    print(f"MatchLite (student) parameters: {s_params:,}")
    print(f"Parameter ratio: {t_params / s_params:.1f}×")

    os.makedirs("checkpoints", exist_ok=True)

    # Phase 1: Teacher training
    if args.mode in ("teacher", "both"):
        print("\n=== Phase 1: MatchLM Teacher Training ===")
        train_teacher(teacher, dataset, num_epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr, device=str(device))
        torch.save(teacher.state_dict(), "checkpoints/matchlm_teacher.pt")
        print("Teacher saved to checkpoints/matchlm_teacher.pt")

    # Phase 2: Student distillation
    if args.mode in ("student", "both"):
        print("\n=== Phase 2: MatchLite Student Distillation ===")
        if args.mode == "student":
            # Load pre-trained teacher if available
            ckpt = "checkpoints/matchlm_teacher.pt"
            if os.path.exists(ckpt):
                teacher.load_state_dict(torch.load(ckpt, map_location="cpu"))
                print(f"Loaded teacher checkpoint from {ckpt}")
        train_student(teacher, student, dataset, num_epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr * 0.5,
                      distill_weight=args.distill_weight, device=str(device))
        torch.save(student.state_dict(), "checkpoints/matchlite_student.pt")
        print("Student saved to checkpoints/matchlite_student.pt")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
