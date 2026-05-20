from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    edge_fpr: float


@torch.no_grad()
def compute_binary_metrics(
    *,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    edge_cohort: torch.Tensor,
) -> BinaryMetrics:
    y_true = y_true.to(torch.int64)
    y_pred = y_pred.to(torch.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    edge_mask = (y_true == 0) & edge_cohort.bool()
    edge_fp = int((edge_mask & (y_pred == 1)).sum().item())
    edge_total = int(edge_mask.sum().item())
    edge_fpr = edge_fp / max(1, edge_total)

    return BinaryMetrics(precision=precision, recall=recall, f1=f1, edge_fpr=edge_fpr)
