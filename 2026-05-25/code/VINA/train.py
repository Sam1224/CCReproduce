"""
VINA Training Script

Joint training on image + video frame pairs with:
  L_VINA = L_CE + λ * L_SCL

L_CE: cross-entropy on both images and video frames (unified classification)
L_SCL: cross-modal supervised contrastive loss (aligns image/video embeddings from same source)

Usage:
    python train.py --epochs 10 --batch_size 32 --lambda_scl 0.5
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ToyAIGCDataset
from model import VINADetector, CrossModalContrastiveLoss


def train_epoch(model, loader, optimizer, ce_loss, scl_loss, lambda_scl, device):
    model.train()
    total_loss = 0.0
    correct_img = 0
    correct_vid = 0
    total = 0

    for images, video_frames, labels in loader:
        images = images.to(device)
        video_frames = video_frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass for images
        logits_img, proj_img = model(images, return_proj=True)

        # Forward pass for video frames (same model, same weights)
        logits_vid, proj_vid = model(video_frames, return_proj=True)

        # Classification loss on both modalities
        loss_ce = ce_loss(logits_img, labels) + ce_loss(logits_vid, labels)

        # Cross-modal supervised contrastive loss
        loss_scl = scl_loss(proj_img, proj_vid, labels)

        # Total VINA loss
        loss = loss_ce + lambda_scl * loss_scl

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct_img += (logits_img.argmax(1) == labels).sum().item()
        correct_vid += (logits_vid.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return {
        'loss': total_loss / total,
        'acc_img': correct_img / total,
        'acc_vid': correct_vid / total,
    }


@torch.no_grad()
def eval_epoch(model, loader, ce_loss, device):
    model.eval()
    total_loss = 0.0
    correct_img = 0
    correct_vid = 0
    total = 0

    for images, video_frames, labels in loader:
        images = images.to(device)
        video_frames = video_frames.to(device)
        labels = labels.to(device)

        logits_img = model(images)
        logits_vid = model(video_frames)

        loss = ce_loss(logits_img, labels) + ce_loss(logits_vid, labels)
        total_loss += loss.item() * labels.size(0)
        correct_img += (logits_img.argmax(1) == labels).sum().item()
        correct_vid += (logits_vid.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return {
        'loss': total_loss / total,
        'acc_img': correct_img / total,
        'acc_vid': correct_vid / total,
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Datasets
    train_ds = ToyAIGCDataset(n_samples=args.n_train, image_size=args.image_size, mode='train')
    val_ds = ToyAIGCDataset(n_samples=args.n_val, image_size=args.image_size, mode='val')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)

    # Model
    model = VINADetector(embed_dim=args.embed_dim, proj_dim=args.proj_dim).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    scl_loss = CrossModalContrastiveLoss(temperature=args.temperature)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs('checkpoints', exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(model, train_loader, optimizer, ce_loss, scl_loss,
                                   args.lambda_scl, device)
        val_stats = eval_epoch(model, val_loader, ce_loss, device)
        scheduler.step()

        avg_val_acc = (val_stats['acc_img'] + val_stats['acc_vid']) / 2
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), 'checkpoints/vina_best.pt')

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss={train_stats['loss']:.4f} img_acc={train_stats['acc_img']:.3f} vid_acc={train_stats['acc_vid']:.3f} | "
            f"Val loss={val_stats['loss']:.4f} img_acc={val_stats['acc_img']:.3f} vid_acc={val_stats['acc_vid']:.3f}"
        )

    print(f"\nBest val acc: {best_val_acc:.4f} — checkpoint: checkpoints/vina_best.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--n_val', type=int, default=400)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--proj_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_scl', type=float, default=0.5,
                        help='Weight for cross-modal contrastive loss (λ in paper)')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature τ for contrastive loss')
    args = parser.parse_args()
    main(args)
