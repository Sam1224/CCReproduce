from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from data import TM20KDataset, build_cfg_from_ckpt, collate_batch
from model import FullAttentionTeacher, ModelConfig, TM20KStudent, TruncatedBaseline
from train import BASELINE_CKPT, STUDENT_CKPT, TEACHER_CKPT, evaluate, measure_latency


def load_model(ckpt_path, cls, device: torch.device):
    payload = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = build_cfg_from_ckpt(payload)
    mcfg = ModelConfig(**payload["model_cfg"])
    model = cls(mcfg).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return payload, cfg, model


def main() -> None:
    if not (os.path.exists(TEACHER_CKPT) and os.path.exists(STUDENT_CKPT) and os.path.exists(BASELINE_CKPT)):
        raise SystemExit("missing checkpoints; run `python train.py` first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_payload, cfg, teacher = load_model(TEACHER_CKPT, FullAttentionTeacher, device)
    _, _, student = load_model(STUDENT_CKPT, TM20KStudent, device)
    _, _, baseline = load_model(BASELINE_CKPT, TruncatedBaseline, device)

    test_ds = TM20KDataset(cfg, split="test")
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_batch)

    t_metrics = evaluate(teacher, test_loader, device)
    s_metrics = evaluate(student, test_loader, device)
    b_metrics = evaluate(baseline, test_loader, device)

    print("=== TM20K toy evaluation ===")
    print("teacher :", {k: round(v, 4) for k, v in t_metrics.items()}, "latency_ms", round(measure_latency(teacher, test_loader, device), 3))
    print("student :", {k: round(v, 4) for k, v in s_metrics.items()}, "latency_ms", round(measure_latency(student, test_loader, device), 3))
    print("baseline:", {k: round(v, 4) for k, v in b_metrics.items()}, "latency_ms", round(measure_latency(baseline, test_loader, device), 3))
    print("expectation: teacher remains strongest, while the merged student recovers more long-range signal than a short-window baseline in this toy setup")


if __name__ == "__main__":
    main()
