"""
Training script for Dual-Path Content Moderation (arXiv 2512.03553)

Training procedure:
  1. Run MockMLLM Teacher offline to annotate training data
     (generates soft labels + teacher embeddings)
  2. Build violation store from MLLM-embedded violation cases
  3. Train student model with:
     - Path1: CE + KL distillation
     - Path2: NT-Xent contrastive + embedding alignment
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, os.path.dirname(__file__))

from model.dual_path import DualPathModerationModel, VIOLATION_CLASSES
from model.mllm_distill import MockMLLMTeacher, DistillationLoss, OfflineMLLMAnnotator
from model.similarity_store import build_toy_violation_store
from data.livestream_dataset import ToyLivestreamDataset, collate_fn


def train(args):
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize models
    model = DualPathModerationModel(
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        embed_dim=args.embed_dim,
    ).to(device)

    teacher = MockMLLMTeacher(num_classes=len(VIOLATION_CLASSES), embed_dim=args.embed_dim).to(device)
    distill_loss_fn = DistillationLoss(temperature=4.0, alpha=0.5, beta=0.5)

    # Build violation store (offline — uses teacher MLLM embeddings)
    print("Building violation store from MLLM embeddings...")
    violation_store = build_toy_violation_store(model, embed_dim=args.embed_dim, num_per_class=20)

    # Dataset
    dataset = ToyLivestreamDataset(num_samples=args.num_samples, violation_rate=0.35)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    # Offline annotation (teacher runs offline — expensive in production)
    print("Running offline MLLM annotation...")
    annotator = OfflineMLLMAnnotator(teacher, batch_size=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs
    )

    best_val_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {"total": 0, "ce": 0, "kl": 0, "embed": 0}

        for step, batch in enumerate(train_loader):
            video = batch["video_frames"].to(device)
            text_ids = batch["text_ids"].to(device)
            audio = batch["audio_mel"].to(device)
            hard_labels = batch["hard_label"].to(device)
            soft_labels = batch["soft_label"].to(device)

            # Forward pass
            outputs = model(video_frames=video, text_ids=text_ids, audio_mel=audio)
            features = outputs["features"]

            # Get teacher embeddings (offline in production; here: live for toy)
            with torch.no_grad():
                teacher_annots = annotator.annotate_batch(features.detach())
                teacher_soft = teacher_annots["soft_labels"]
                teacher_embeds = teacher_annots["teacher_embeds"]

            # Path 1 + Path 2 losses
            task_loss = model.compute_loss(
                outputs,
                hard_labels=hard_labels,
                mllm_soft_labels=teacher_soft,
                mllm_embeds=teacher_embeds,
            )

            # Combined loss
            loss = task_loss["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses["total"] += loss.item()
            epoch_losses["ce"] += task_loss["path1_ce_loss"].item()
            epoch_losses["kl"] += task_loss["path1_kl_loss"].item()
            epoch_losses["embed"] += task_loss["embed_align_loss"].item()

            if step % 10 == 0:
                print(
                    f"  Epoch {epoch+1} Step {step+1}/{len(train_loader)} "
                    f"Loss={loss.item():.4f} CE={task_loss['path1_ce_loss'].item():.4f} "
                    f"KL={task_loss['path1_kl_loss'].item():.4f}"
                )

        n_steps = len(train_loader)
        print(
            f"\nEpoch {epoch+1} | Total={epoch_losses['total']/n_steps:.4f} "
            f"CE={epoch_losses['ce']/n_steps:.4f} "
            f"KL={epoch_losses['kl']/n_steps:.4f} "
            f"Embed={epoch_losses['embed']/n_steps:.4f}"
        )

        # Validation
        val_metrics = evaluate(model, val_loader, violation_store, device)
        print(
            f"  Val | Acc={val_metrics['accuracy']:.4f} "
            f"F1={val_metrics['macro_f1']:.4f} "
            f"Recall@P80={val_metrics.get('recall_at_p80', 0):.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            ckpt_path = os.path.join(args.output_dir, "dualpath.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best checkpoint: {ckpt_path}")

    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")
    return model


def evaluate(model, loader, violation_store, device):
    """Evaluation: accuracy, F1, and recall@80% precision for each path."""
    model.eval()
    all_preds, all_labels, all_p1_scores = [], [], []

    with torch.no_grad():
        for batch in loader:
            video = batch["video_frames"].to(device)
            text_ids = batch["text_ids"].to(device)
            audio = batch["audio_mel"].to(device)
            labels = batch["hard_label"]

            result = model.predict(
                video_frames=video, text_ids=text_ids, audio_mel=audio,
                violation_store=violation_store,
            )

            preds = (result["path1_scores"] > 0.5).long().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_p1_scores.extend(result["path1_scores"].cpu().tolist())

    # Accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0.0

    # Macro F1
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    is_violation = lambda lbl: lbl != 0
    for p, l in zip(all_preds, all_labels):
        if p == 1 and is_violation(l):
            tp["violation"] += 1
        elif p == 1 and not is_violation(l):
            fp["violation"] += 1
        elif p == 0 and is_violation(l):
            fn["violation"] += 1

    prec = tp["violation"] / (tp["violation"] + fp["violation"] + 1e-8)
    rec = tp["violation"] / (tp["violation"] + fn["violation"] + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    # Recall @ 80% precision (paper's key metric)
    thresholds = sorted(set(all_p1_scores), reverse=True)
    recall_at_p80 = 0.0
    for thr in thresholds:
        preds_t = [1 if s > thr else 0 for s in all_p1_scores]
        tp_t = sum(p == 1 and is_violation(l) for p, l in zip(preds_t, all_labels))
        fp_t = sum(p == 1 and not is_violation(l) for p, l in zip(preds_t, all_labels))
        fn_t = sum(p == 0 and is_violation(l) for p, l in zip(preds_t, all_labels))
        p_t = tp_t / (tp_t + fp_t + 1e-8)
        r_t = tp_t / (tp_t + fn_t + 1e-8)
        if p_t >= 0.8:
            recall_at_p80 = r_t
            break

    return {
        "accuracy": accuracy,
        "macro_f1": f1,
        "recall_at_p80": recall_at_p80,
        "precision": prec,
        "recall": rec,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    model = train(args)
    print("Done.")
