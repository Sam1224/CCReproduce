from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvalMetrics:
    acc: float
    macro_f1: float
    win_rate: float


def _macro_f1(tp_fp_fn_by_class):
    f1s = []
    for tp, fp, fn in tp_fp_fn_by_class:
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        f1s.append(f1)
    return float(sum(f1s) / len(f1s))


@torch.no_grad()
def evaluate(model, ds, batch_size: int = 128) -> EvalMetrics:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    tp_fp_fn = [[0, 0, 0] for _ in range(4)]

    # "online-like" win-rate proxy: correct@{y>=2} + correct@{y<=1} / total
    win = 0

    for batch in loader:
        out: Dict[str, torch.Tensor] = model(batch["q_ids"], batch["d_ids"])
        pred = out["coarse_logits"].argmax(dim=-1)
        gold = batch["y"]

        total += int(gold.numel())
        correct += int((pred == gold).sum().item())

        for c in range(4):
            pc = pred == c
            gc = gold == c
            tp_fp_fn[c][0] += int((pc & gc).sum().item())
            tp_fp_fn[c][1] += int((pc & ~gc).sum().item())
            tp_fp_fn[c][2] += int((~pc & gc).sum().item())

        is_good = gold >= 2
        win += int(((pred >= 2) == is_good).sum().item())

    return EvalMetrics(
        acc=correct / max(1, total),
        macro_f1=_macro_f1(tp_fp_fn),
        win_rate=win / max(1, total),
    )
