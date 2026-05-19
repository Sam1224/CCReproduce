from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def symmetric_contrastive_loss(query_emb: torch.Tensor, item_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (query_emb @ item_emb.t()) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_q2i = F.cross_entropy(logits, labels)
    loss_i2q = F.cross_entropy(logits.t(), labels)
    return (loss_q2i + loss_i2q) / 2


def spatial_relational_distillation(patch_attn: torch.Tensor, gt_box_mask: torch.Tensor) -> torch.Tensor:
    """Distill student patch attention towards teacher region mask.

    - patch_attn: (B, N), values sum to 1 (from attention softmax)
    - gt_box_mask: (B, N), in {0,1}

    We normalize gt_box_mask to a distribution and minimize KL(teacher || student).
    """

    teacher = gt_box_mask.float()
    teacher = teacher / (teacher.sum(dim=1, keepdim=True) + 1e-6)
    student = patch_attn.clamp_min(1e-8)
    teacher = teacher.clamp_min(1e-8)
    return torch.mean(torch.sum(teacher * (teacher.log() - student.log()), dim=1))


def similarity_distribution_distillation(
    query_emb: torch.Tensor,
    student_item_emb: torch.Tensor,
    teacher_item_emb: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Distill similarity distributions within a batch."""

    s_student = (query_emb @ student_item_emb.t()) / temperature
    s_teacher = (query_emb @ teacher_item_emb.t()) / temperature

    p_teacher = F.softmax(s_teacher, dim=1).clamp_min(1e-8)
    p_student = F.softmax(s_student, dim=1).clamp_min(1e-8)

    return torch.mean(torch.sum(p_teacher * (p_teacher.log() - p_student.log()), dim=1))


def tiger_fg_loss(
    *,
    query_emb: torch.Tensor,
    item_emb: torch.Tensor,
    patch_attn: torch.Tensor,
    gt_box_mask: torch.Tensor,
    temperature: float = 0.07,
    teacher_item_emb: Optional[torch.Tensor] = None,
    w_contrast: float = 1.0,
    w_spatial: float = 0.2,
    w_simdist: float = 0.2,
) -> torch.Tensor:
    loss = w_contrast * symmetric_contrastive_loss(query_emb, item_emb, temperature)

    if w_spatial > 0:
        loss = loss + w_spatial * spatial_relational_distillation(patch_attn, gt_box_mask)

    if w_simdist > 0 and teacher_item_emb is not None:
        loss = loss + w_simdist * similarity_distribution_distillation(
            query_emb=query_emb,
            student_item_emb=item_emb,
            teacher_item_emb=teacher_item_emb,
            temperature=temperature,
        )

    return loss
