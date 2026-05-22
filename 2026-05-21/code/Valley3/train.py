"""Training script for Valley3 (toy reproduction of arXiv:2605.01278).

Demonstrates the four-stage training concept by cycling through stages.

Usage:
    python train.py --epochs 10 --output_dir runs/valley3
"""
import argparse
import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from collections import defaultdict
from valley3 import Valley3Model, OmniECommerceDataset, ECommerceLoss, compute_task_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_enc", type=int, default=64)
    p.add_argument("--d_llm", type=int, default=128)
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--output_dir", type=str, default="runs/valley3")
    return p.parse_args()


def collate_fn(batch):
    """Collate items — group by task."""
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        if k == "task":
            collated[k] = [b[k] for b in batch]
        elif k == "label":
            # Labels have different shapes for captioning vs. others
            labels = [b[k] for b in batch]
            try:
                collated[k] = torch.stack(labels)
            except RuntimeError:
                # Different sizes (captioning vs. cls) — pad to max len
                max_len = max(l.numel() for l in labels)
                padded = []
                for l in labels:
                    if l.dim() == 0:
                        padded.append(l.unsqueeze(0).expand(max_len))
                    else:
                        p = torch.zeros(max_len, dtype=l.dtype)
                        p[:l.numel()] = l
                        padded.append(p)
                collated[k] = torch.stack(padded)
        else:
            collated[k] = torch.stack([b[k] for b in batch])
    return collated


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    task_accs = defaultdict(list)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            task = batch["task"]
            text = batch["text"].to(device)
            image = batch["image"].to(device)
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            # Fix captioning label shape
            task_name = task[0] if isinstance(task, list) else task
            if task_name == "captioning" and labels.dim() == 1:
                labels = labels.unsqueeze(1)

            logits = model(task, text, image, video, audio)
            loss = criterion(logits, labels, task)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            metrics = compute_task_metrics(logits.detach(), labels, task)
            for k, v in metrics.items():
                task_accs[f"{task_name}_{k}"].append(v)

    avg_loss = total_loss / len(loader)
    avg_metrics = {k: sum(v) / len(v) for k, v in task_accs.items()}
    return avg_loss, avg_metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Dataset (each sample has a single task type for batching simplicity)
    # For single-task batching, create separate per-task datasets
    from valley3.dataset import TASK_TYPES
    from torch.utils.data import ConcatDataset

    ds = OmniECommerceDataset(num_samples=args.num_samples)
    n_train = int(0.8 * len(ds))
    train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train])

    # Note: mixing tasks in batch uses collate_fn; captioning batches may have shape issues
    # For simplicity, filter per task at iteration time
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn)

    model = Valley3Model(d_enc=args.d_enc, d_llm=args.d_llm).to(device)

    # Demonstrate 4-stage training: run a few epochs per stage
    stages = ["audio_align", "cross_modal", "ecom_domain", "full"]
    epochs_per_stage = max(1, args.epochs // len(stages))

    criterion = ECommerceLoss()
    history = []

    for stage_i, stage in enumerate(stages):
        model.set_stage(stage)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n=== Stage {stage_i+1}: {stage} | trainable params: {trainable:,} ===")
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-4,
        )

        n_epochs = epochs_per_stage if stage_i < len(stages) - 1 else \
                   args.epochs - epochs_per_stage * (len(stages) - 1)

        for ep in range(1, max(n_epochs, 1) + 1):
            tr_loss, tr_metrics = run_epoch(model, train_dl, criterion, optimizer, device, True)
            va_loss, va_metrics = run_epoch(model, val_dl, criterion, optimizer, device, False)
            entry = {"stage": stage, "epoch": ep,
                     "train_loss": tr_loss, "val_loss": va_loss,
                     **{f"val_{k}": v for k, v in va_metrics.items()}}
            history.append(entry)
            print(f"  Ep {ep} | tr_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
                  f"| {', '.join(f'{k}={v:.3f}' for k,v in va_metrics.items())}")

    ckpt_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved → {ckpt_path}")


if __name__ == "__main__":
    main()
