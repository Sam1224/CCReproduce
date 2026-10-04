import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data import FeatureConfig, ToyLivestreamDataset, build_reference_bank
from model import ModelConfig, MultimodalStudent, FrozenMLLMTeacher, contrastive_reference_loss, distillation_loss


def train(args):
    torch.manual_seed(args.seed)
    data_config = FeatureConfig(seed=args.seed)
    dataset = ToyLivestreamDataset(size=args.samples, config=data_config)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model_config = ModelConfig(
        visual_dim=data_config.visual_dim,
        audio_dim=data_config.audio_dim,
        text_dim=data_config.text_dim,
        num_classes=data_config.num_classes,
    )
    student = MultimodalStudent(model_config)
    teacher = FrozenMLLMTeacher(model_config)
    reference_bank = build_reference_bank(dataset)
    ref_batch = {
        "visual": reference_bank["raw"][:, :data_config.visual_dim],
        "audio": reference_bank["raw"][:, data_config.visual_dim:data_config.visual_dim + data_config.audio_dim],
        "text": reference_bank["raw"][:, data_config.visual_dim + data_config.audio_dim:],
    }
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-3)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        student.train()
        running = 0.0
        for batch in train_loader:
            with torch.no_grad():
                teacher_out = teacher(batch)
                reference_embeddings = teacher(ref_batch)["embedding"]
            student_out = student(batch)
            scores = student.score_references(student_out["embedding"], reference_embeddings)
            cls_loss = F.cross_entropy(student_out["logits"], batch["label"])
            ref_loss = contrastive_reference_loss(scores, batch["reference_id"])
            kd_loss = distillation_loss(student_out, teacher_out)
            loss = cls_loss + args.reference_weight * ref_loss + args.distill_weight * kd_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            running += loss.item()

        val_acc = evaluate_accuracy(student, val_loader)
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": student.state_dict(), "config": model_config.__dict__, "data_config": data_config.__dict__}, out_dir / "best.pt")
        print(f"epoch={epoch} train_loss={running / max(1, len(train_loader)):.4f} val_acc={val_acc:.4f}")


def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch)["logits"]
            correct += (logits.argmax(dim=-1) == batch["label"]).sum().item()
            total += batch["label"].numel()
    return correct / max(1, total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--distill-weight", type=float, default=0.35)
    parser.add_argument("--reference-weight", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="runs/demo")
    train(parser.parse_args())
