import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset import ECommerceSequenceConfig, SyntheticECommerceDataset
from model import FullAttentionRanker, TM20KConfig, distillation_loss


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            target = batch["target"].to(device)
            label = batch["label"].to(device)
            logits = model(sequence, target)
            loss_sum += torch.nn.functional.binary_cross_entropy_with_logits(logits, label).item()
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == label).sum().item()
            total += label.numel()
    return {"loss": loss_sum / max(len(loader), 1), "accuracy": correct / max(total, 1)}


def train_epoch(model, loader, optimizer, device, teacher=None):
    model.train()
    if teacher is not None:
        teacher.eval()
    total_loss = 0.0
    for batch in loader:
        sequence = batch["sequence"].to(device)
        target = batch["target"].to(device)
        label = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(sequence, target)
        if teacher is None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label)
        else:
            with torch.no_grad():
                teacher_logits = teacher(sequence, target)
            loss = distillation_loss(logits, teacher_logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description="Train a compact TM20K reproduction on toy e-commerce data.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dataset-size", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--merged-len", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = ECommerceSequenceConfig(max_seq_len=args.max_seq_len)
    dataset = SyntheticECommerceDataset(size=args.dataset_size, config=data_cfg)
    train_size = int(len(dataset) * 0.85)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(7))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model_cfg = TM20KConfig(max_seq_len=args.max_seq_len, merged_len=args.merged_len)
    teacher = FullAttentionRanker(model_cfg, use_token_merge=False).to(device)
    student = FullAttentionRanker(model_cfg, use_token_merge=True).to(device)

    teacher_optim = torch.optim.AdamW(teacher.parameters(), lr=3e-4, weight_decay=1e-4)
    student_optim = torch.optim.AdamW(student.parameters(), lr=4e-4, weight_decay=1e-4)

    for epoch in range(args.epochs):
        teacher_loss = train_epoch(teacher, train_loader, teacher_optim, device)
        teacher_eval = evaluate(teacher, val_loader, device)
        print(f"teacher epoch={epoch + 1} loss={teacher_loss:.4f} val={teacher_eval}")

    for epoch in range(args.epochs):
        student_loss = train_epoch(student, train_loader, student_optim, device, teacher=teacher)
        student_eval = evaluate(student, val_loader, device)
        print(f"student epoch={epoch + 1} kd_loss={student_loss:.4f} val={student_eval}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model_cfg.__dict__, "model": student.state_dict()}, output_dir / "tm20k_student.pt")
    torch.save({"config": model_cfg.__dict__, "model": teacher.state_dict()}, output_dir / "tm20k_teacher.pt")


if __name__ == "__main__":
    main()
