#!/usr/bin/env python
"""Train the classification path of LivestreamMod."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn.functional as F
from src.dataset import get_loaders
from src.models import ClassificationPath

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = get_loaders(args.data_dir, batch_size=args.batch_size)
    model = ClassificationPath(input_dim=384, num_classes=5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = total_correct = total = 0
        for batch in train_loader:
            fused = batch["fused"].to(device)
            labels = batch["label"].to(device)
            logits = model(fused)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        print(f"Epoch {epoch+1}/{args.epochs} | loss={total_loss/len(train_loader):.4f} "
              f"| acc={total_correct/total:.3f}")

    # Evaluate
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for batch in val_loader:
            fused = batch["fused"].to(device)
            is_viol = batch["is_violation"].to(device)
            logits = model(fused)
            pred_viol = (logits.argmax(1) > 0).float()
            tp += ((pred_viol == 1) & (is_viol == 1)).sum().item()
            fp += ((pred_viol == 1) & (is_viol == 0)).sum().item()
            fn += ((pred_viol == 0) & (is_viol == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    print(f"\nClassification Path Eval — Precision: {precision:.3f}, Recall: {recall:.3f}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "cls_path.pt"))
    print(f"Saved → {args.ckpt_dir}/cls_path.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/toy_cases")
    parser.add_argument("--ckpt_dir", default="data/ckpt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    train(parser.parse_args())
