from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import make_dataset, split
from model import MMClassifier, entropy_from_logits, kd_loss, weighted_kd_loss


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def train_teacher(train, test, epochs: int, device: torch.device) -> MMClassifier:
    teacher = MMClassifier(hidden=256).to(device)
    opt = torch.optim.AdamW(teacher.parameters(), lr=3e-3)

    for ep in range(epochs):
        teacher.train()
        logits = teacher(train.img.to(device), train.txt.to(device))
        loss = F.cross_entropy(logits, train.y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        teacher.eval()
        with torch.no_grad():
            acc = accuracy(teacher(test.img.to(device), test.txt.to(device)), test.y.to(device))
        print(f"teacher epoch={ep} loss={loss.item():.4f} acc={acc:.3f}")

    return teacher


def train_student(mode: str, teacher: MMClassifier, train, test, epochs: int, device: torch.device) -> MMClassifier:
    student = MMClassifier(hidden=96).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-3)

    teacher.eval()
    for ep in range(epochs):
        student.train()
        with torch.no_grad():
            t_logits = teacher(train.img.to(device), train.txt.to(device))
            unc = entropy_from_logits(t_logits)
            # higher entropy => lower weight
            weights = (1.0 / (unc + 1e-3)).detach()

        s_logits = student(train.img.to(device), train.txt.to(device))
        ce = F.cross_entropy(s_logits, train.y.to(device))

        if mode == "uakd":
            kd = weighted_kd_loss(s_logits, t_logits, weights)
        else:
            kd = kd_loss(s_logits, t_logits)

        loss = 0.5 * ce + 0.5 * kd

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        student.eval()
        with torch.no_grad():
            acc = accuracy(student(test.img.to(device), test.txt.to(device)), test.y.to(device))
        print(f"student[{mode}] epoch={ep} loss={loss.item():.4f} acc={acc:.3f}")

    return student


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-teacher", type=int, default=3)
    ap.add_argument("--epochs-student", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = make_dataset(n=3000, seed=1)
    train, test = split(batch, 0.8)

    teacher = train_teacher(train, test, epochs=args.epochs_teacher, device=device)

    vanilla = train_student("vanilla", teacher, train, test, epochs=args.epochs_student, device=device)
    uakd = train_student("uakd", teacher, train, test, epochs=args.epochs_student, device=device)

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"state_dict": teacher.state_dict()}, ckpt / "teacher.pt")
    torch.save({"state_dict": vanilla.state_dict()}, ckpt / "student_vanilla.pt")
    torch.save({"state_dict": uakd.state_dict()}, ckpt / "student_uakd.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
