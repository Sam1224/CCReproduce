from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import DataConfig, TM20KDataset, binary_auc, build_cfg_from_ckpt, collate_batch
from model import FullAttentionTeacher, ModelConfig, TM20KStudent, TruncatedBaseline

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
TEACHER_CKPT = CKPT_DIR / "teacher.pt"
STUDENT_CKPT = CKPT_DIR / "tm20k_student.pt"
BASELINE_CKPT = CKPT_DIR / "truncated_baseline.pt"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    probs_all = []
    labels_all = []
    for batch in loader:
        seq = batch["seq"].to(device)
        mask = batch["mask"].to(device)
        ad_cat = batch["ad_cat"].to(device)
        label = batch["label"].to(device)
        prob = torch.sigmoid(model(seq, mask, ad_cat))
        probs_all.append(prob)
        labels_all.append(label)
    probs = torch.cat(probs_all)
    labels = torch.cat(labels_all)
    preds = (probs >= 0.5).float()
    acc = float((preds == labels).float().mean().item())
    return {"auc": binary_auc(probs, labels), "acc": acc}


@torch.no_grad()
def measure_latency(model, loader, device: torch.device, warmup: int = 2, steps: int = 8) -> float:
    model.eval()
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= warmup + steps:
            break
    if not batches:
        return 0.0
    times = []
    for idx, batch in enumerate(batches):
        seq = batch["seq"].to(device)
        mask = batch["mask"].to(device)
        ad_cat = batch["ad_cat"].to(device)
        start = time.perf_counter()
        _ = model(seq, mask, ad_cat)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000.0
        if idx >= warmup:
            times.append(elapsed)
    return float(sum(times) / max(1, len(times)))


def train_supervised(model, train_loader, valid_loader, device: torch.device, epochs: int, lr: float):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best_metric = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for batch in train_loader:
            seq = batch["seq"].to(device)
            mask = batch["mask"].to(device)
            ad_cat = batch["ad_cat"].to(device)
            label = batch["label"].to(device)
            logit = model(seq, mask, ad_cat)
            loss = F.binary_cross_entropy_with_logits(logit, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            n += 1
        valid = evaluate(model, valid_loader, device)
        print(f"epoch={epoch:02d} loss={loss_sum / max(1, n):.4f} valid_auc={valid['auc']:.4f} valid_acc={valid['acc']:.4f}")
        if valid["auc"] > best_metric:
            best_metric = valid["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)


def train_student(student, teacher, train_loader, valid_loader, device: torch.device, epochs: int, lr: float):
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-3)
    best_metric = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        student.train()
        loss_sum = 0.0
        n = 0
        for batch in train_loader:
            seq = batch["seq"].to(device)
            mask = batch["mask"].to(device)
            ad_cat = batch["ad_cat"].to(device)
            label = batch["label"].to(device)
            with torch.no_grad():
                teacher_logit = teacher(seq, mask, ad_cat)
            student_logit = student(seq, mask, ad_cat)
            loss_sup = F.binary_cross_entropy_with_logits(student_logit, label)
            loss_kd = F.mse_loss(torch.sigmoid(student_logit), torch.sigmoid(teacher_logit))
            loss = loss_sup + 0.4 * loss_kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            n += 1
        valid = evaluate(student, valid_loader, device)
        print(f"student epoch={epoch:02d} loss={loss_sum / max(1, n):.4f} valid_auc={valid['auc']:.4f} valid_acc={valid['acc']:.4f}")
        if valid["auc"] > best_metric:
            best_metric = valid["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    if best_state is not None:
        student.load_state_dict(best_state)


def save_checkpoint(path: Path, model, cfg: DataConfig, mcfg: ModelConfig, tag: str) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"tag": tag, "data_cfg": cfg.__dict__, "model_cfg": mcfg.__dict__, "state_dict": model.state_dict()}, path)
    print(f"saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs-teacher", type=int, default=4)
    parser.add_argument("--epochs-student", type=int, default=4)
    parser.add_argument("--epochs-baseline", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cfg = DataConfig(seed=args.seed)
    train_ds = TM20KDataset(cfg, split="train")
    valid_ds = TM20KDataset(cfg, split="valid")
    test_ds = TM20KDataset(cfg, split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    mcfg = ModelConfig(num_items=cfg.num_items, num_categories=cfg.num_categories, seq_len=cfg.seq_len)

    teacher = FullAttentionTeacher(mcfg).to(device)
    print("== train teacher ==")
    train_supervised(teacher, train_loader, valid_loader, device, epochs=args.epochs_teacher, lr=args.lr)
    print("teacher test:", evaluate(teacher, test_loader, device), "latency_ms=", round(measure_latency(teacher, test_loader, device), 3))
    save_checkpoint(TEACHER_CKPT, teacher, cfg, mcfg, "teacher")

    baseline = TruncatedBaseline(mcfg).to(device)
    print("== train truncated baseline ==")
    train_supervised(baseline, train_loader, valid_loader, device, epochs=args.epochs_baseline, lr=args.lr)
    print("baseline test:", evaluate(baseline, test_loader, device), "latency_ms=", round(measure_latency(baseline, test_loader, device), 3))
    save_checkpoint(BASELINE_CKPT, baseline, cfg, mcfg, "baseline")

    student = TM20KStudent(mcfg).to(device)
    print("== train tm20k student ==")
    train_student(student, teacher, train_loader, valid_loader, device, epochs=args.epochs_student, lr=args.lr)
    print("student test:", evaluate(student, test_loader, device), "latency_ms=", round(measure_latency(student, test_loader, device), 3))
    save_checkpoint(STUDENT_CKPT, student, cfg, mcfg, "student")


if __name__ == "__main__":
    main()
