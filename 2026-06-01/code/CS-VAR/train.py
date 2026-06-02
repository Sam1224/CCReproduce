"""
Training script for CS-VAR.

Two-phase training (paper §3.3):
  Phase 1: Build session history database from training data.
  Phase 2: Train student model with KD from LLM teacher.
"""
import argparse
import os
import torch
import torch.optim as optim

from model import SessionRiskModel
from llm_teacher import LLMTeacherSurrogate, CSVARDistillationLoss
from retrieval import SessionDatabase
from dataset import get_dataloaders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--event_dim", type=int, default=64)
    p.add_argument("--num_risk_levels", type=int, default=3)
    p.add_argument("--k_retrieved", type=int, default=5)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    return p.parse_args()


def build_history_database(model, history_loader, db, device):
    """Pre-populate session database with training history embeddings."""
    model.eval()
    with torch.no_grad():
        for batch in history_loader:
            events = batch["events"].to(device)
            labels = batch["risk_label"]
            emb = model.encode_session(events)  # (B, H)
            db.add(emb, labels)
    db.build_index()
    print(f"History database built with {len(db.risk_labels)} sessions.")


def train_epoch(model, teacher, loader, optimizer, loss_fn, db, k, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        events = batch["events"].to(device)
        labels = batch["risk_label"].to(device)

        # Step 1: Encode current sessions (student)
        with torch.no_grad():
            curr_emb = model.encode_session(events)

        # Step 2: Retrieve cross-session evidence from history database
        retrieved_emb, _ = db.retrieve(curr_emb, k=k)
        retrieved_emb = retrieved_emb.to(device)

        # Step 3: LLM teacher generates soft labels
        teacher.eval()
        with torch.no_grad():
            teacher_logits = teacher(curr_emb, retrieved_emb)

        # Step 4: Student forward pass
        optimizer.zero_grad()
        student_logits, emb = model(events, retrieved_emb)

        # Step 5: Distillation loss
        loss = loss_fn(student_logits, teacher_logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, db, k, device):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        events = batch["events"].to(device)
        labels = batch["risk_label"].to(device)
        curr_emb = model.encode_session(events)
        retrieved_emb, _ = db.retrieve(curr_emb, k=k)
        retrieved_emb = retrieved_emb.to(device)
        logits, _ = model(events, retrieved_emb)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, test_loader, history_loader = get_dataloaders(
        args.batch_size, args.event_dim, 64, args.num_risk_levels
    )

    model = SessionRiskModel(
        event_dim=args.event_dim,
        hidden_dim=args.hidden_dim,
        num_risk_levels=args.num_risk_levels,
    ).to(device)

    teacher = LLMTeacherSurrogate(
        hidden_dim=args.hidden_dim,
        k=args.k_retrieved,
        num_risk_levels=args.num_risk_levels,
    ).to(device)

    db = SessionDatabase(args.hidden_dim)
    loss_fn = CSVARDistillationLoss()

    # Phase 1: Build history database
    print("Phase 1: Building session history database...")
    build_history_database(model, history_loader, db, device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, teacher, train_loader, optimizer, loss_fn, db, args.k_retrieved, device
        )
        val_acc = evaluate(model, val_loader, db, args.k_retrieved, device)
        scheduler.step()
        print(f"Epoch {epoch:3d} | Loss {train_loss:.4f} | Val Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{args.save_dir}/best.pt")
            torch.save(db, f"{args.save_dir}/db.pt")

    test_acc = evaluate(model, test_loader, db, args.k_retrieved, device)
    print(f"\nTest Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
