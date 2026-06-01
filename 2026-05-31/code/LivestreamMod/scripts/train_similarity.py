#!/usr/bin/env python
"""Train the similarity path with MLLM teacher distillation."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn.functional as F
from src.dataset import get_loaders
from src.models import SimilarityPath, ClassificationPath
from src.distillation import TeacherMLP, DistillationLoss, pretrain_teacher

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = get_loaders(args.data_dir, batch_size=args.batch_size)

    # Step 1: Pre-train teacher MLP (proxy for MLLM)
    print("Pre-training teacher (proxy for MLLM)...")
    teacher = TeacherMLP(input_dim=384, num_classes=5)
    teacher = pretrain_teacher(teacher, train_loader, epochs=15, device=device)
    teacher.eval()

    # Step 2: Train student similarity encoder with distillation
    print("\nTraining student similarity encoder via distillation...")
    student_cls = ClassificationPath(384, 192, 5).to(device)
    sim_path = SimilarityPath(modal_dim=128, embed_dim=64).to(device)
    dist_loss_fn = DistillationLoss(temperature=3.0, lam=0.7)
    opt = torch.optim.Adam(
        list(student_cls.parameters()) + list(sim_path.parameters()),
        lr=1e-3, weight_decay=1e-4
    )

    for epoch in range(args.epochs):
        student_cls.train(); sim_path.train()
        total_loss = 0.0
        for batch in train_loader:
            fused = batch["fused"].to(device)
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            labels = batch["label"].to(device)

            # Teacher soft labels
            with torch.no_grad():
                soft = teacher.soft_labels(fused, temperature=3.0)

            # Student classification logits
            student_logits = student_cls(fused)

            # Distillation loss
            loss = dist_loss_fn(student_logits, soft, labels)

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs} | distil_loss={total_loss/len(train_loader):.4f}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save(sim_path.state_dict(), os.path.join(args.ckpt_dir, "sim_path.pt"))
    print(f"Saved → {args.ckpt_dir}/sim_path.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/toy_cases")
    parser.add_argument("--ckpt_dir", default="data/ckpt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    train(parser.parse_args())
