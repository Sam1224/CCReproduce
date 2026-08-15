import argparse
import torch
from torch.utils.data import DataLoader

from data import ToyEcommerceSequenceDataset
from tm20k import TM20KEncoder, TM20KRanker, TM20KStudent, TokenMerger, tm20k_distillation_loss


def run_epoch(model, loader, optimizer=None, teacher=None, device="cpu"):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        categories = batch["categories"].to(device)
        positions = batch["positions"].to(device)
        labels = batch["label"].to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        output = model(tokens, categories, positions)
        if teacher is None:
            loss = torch.nn.functional.cross_entropy(output["logits"], labels)
        else:
            with torch.no_grad():
                teacher_output = teacher(tokens, categories, positions)
            loss = tm20k_distillation_loss(output, teacher_output, labels)
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (output["logits"].argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return {"loss": total_loss / total, "accuracy": correct / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--merge-ratio", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = ToyEcommerceSequenceDataset(seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    teacher = TM20KRanker(TM20KEncoder(max_seq_len=args.seq_len)).to(args.device)
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=3e-4)
    for epoch in range(args.epochs):
        metrics = run_epoch(teacher, loader, teacher_optimizer, device=args.device)
        print(f"teacher epoch={epoch + 1} loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f}")

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()

    student_ranker = TM20KRanker(TM20KEncoder(max_seq_len=args.seq_len)).to(args.device)
    student = TM20KStudent(student_ranker, TokenMerger(strategy="recent_keep", merge_ratio=args.merge_ratio)).to(args.device)
    student_optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
    for epoch in range(args.epochs):
        metrics = run_epoch(student, loader, student_optimizer, teacher=teacher, device=args.device)
        print(f"student epoch={epoch + 1} loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f}")

    torch.save({"teacher": teacher.state_dict(), "student": student.state_dict(), "args": vars(args)}, "tm20k_toy_checkpoint.pt")


if __name__ == "__main__":
    main()
