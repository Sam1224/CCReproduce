"""
Training script for HybridMod.

Training procedure (paper §3):
  Phase 1: Populate reference gallery using MLLM teacher embeddings.
  Phase 2: Train Pipeline A (classifier) with KD from teacher.
  Phase 3: Train Pipeline B (similarity re-ranker) jointly.
  Phase 4: Fine-tune fusion weights.
"""
import argparse
import torch
import torch.optim as optim
import os

from model import HybridMod
from mllm_teacher import MLLMTeacherSurrogate, KnowledgeDistillationLoss, build_reference_gallery
from dataset import get_dataloaders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    return p.parse_args()


def train_epoch(model, teacher, loader, optimizer, kd_loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        text = batch["text_feat"].to(device)
        audio = batch["audio_feat"].to(device)
        visual = batch["visual_feat"].to(device)
        labels = batch["label"].to(device)

        # Teacher soft labels (no grad)
        teacher.eval()
        with torch.no_grad():
            teacher_logits, _ = teacher(text, visual)

        optimizer.zero_grad()
        out = model(text, audio, visual)

        # Distillation loss on Pipeline A (main classifier)
        loss, ce, kd = kd_loss_fn(out["logits_a"], teacher_logits, labels)

        # Standard CE on fused logits
        fused_ce = torch.nn.functional.cross_entropy(out["logits"], labels)
        total = loss + 0.5 * fused_ce

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += total.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        text = batch["text_feat"].to(device)
        audio = batch["audio_feat"].to(device)
        visual = batch["visual_feat"].to(device)
        labels = batch["label"].to(device)
        out = model(text, audio, visual)
        preds = out["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        args.batch_size, args.num_classes
    )

    model = HybridMod(
        text_dim=768,
        audio_dim=128,
        visual_dim=2048,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    ).to(device)

    teacher = MLLMTeacherSurrogate(
        text_dim=768,
        visual_dim=2048,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    ).to(device)

    kd_loss_fn = KnowledgeDistillationLoss(
        temperature=args.temperature, alpha=args.alpha
    )

    # Phase 1: Populate reference gallery
    print("Phase 1: Populating reference gallery with teacher embeddings...")
    build_reference_gallery(teacher, train_loader, model, device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, teacher, train_loader, optimizer, kd_loss_fn, device
        )
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:3d} | Loss {train_loss:.4f} | Val Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{args.save_dir}/best.pt")

    test_acc = evaluate(model, test_loader, device)
    print(f"\nTest Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
