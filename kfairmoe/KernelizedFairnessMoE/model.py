from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def grad_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradReverse.apply(x, scale)


class KernelINLPProjector(nn.Module):
    """Closed-form projector (toy).

    The paper describes an optimization-free kernelized INLP projector.
    Here we implement a simple closed-form *linear* null-space projector:

      P = I - W^T (W W^T + eps I)^{-1} W

    where W is a sensitive subspace basis (toy: fixed random basis).

    This is sufficient to make the flow correct and trainable end-to-end.
    """

    def __init__(self, d_model: int, subspace_dim: int = 16, eps: float = 1e-4, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)

        w = torch.randn(subspace_dim, d_model, generator=g)
        ww_t = w @ w.t()  # (k,k)
        inv = torch.inverse(ww_t + eps * torch.eye(subspace_dim))
        p = torch.eye(d_model) - w.t() @ inv @ w  # (d,d)

        self.register_buffer("P", p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,d)
        return torch.einsum("bld,dd->bld", x, self.P)


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedMoEAdapter(nn.Module):
    def __init__(self, d_model: int, num_experts: int = 4, hidden_dim: int = 256) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([ExpertFFN(d_model, hidden_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,d)
        gate_scores = F.softmax(self.gate(x), dim=-1)  # (B,L,E)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)  # (B,L,E,d)
        moe = torch.sum(gate_scores.unsqueeze(-1) * expert_outputs, dim=-2)  # (B,L,d)
        return x + moe


class LowRankAdapter(nn.Module):
    def __init__(self, d_model: int, rank: int = 32) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(x)))


@dataclass
class FairnessOutput:
    task_logits: torch.Tensor  # (B,L,num_items)
    sensitive_logits: torch.Tensor  # (B,L,num_sensitive)


class OneModel(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_items: int = 200,
        num_sensitive: int = 2,
        num_experts: int = 4,
        fairness_grl_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_items = num_items
        self.num_sensitive = num_sensitive
        self.fairness_grl_scale = fairness_grl_scale

        self.projector = KernelINLPProjector(d_model=d_model)
        self.moe = GatedMoEAdapter(d_model=d_model, num_experts=num_experts)
        self.attr_adapters = nn.ModuleList([LowRankAdapter(d_model) for _ in range(num_sensitive)])

        self.task_head = nn.Linear(d_model, num_items)
        self.sensitive_head = nn.Linear(d_model, num_sensitive)

    def freeze_modules(self) -> None:
        # Projector is closed-form and kept frozen.
        for p in self.projector.parameters():
            p.requires_grad_(False)

    def get_optim(self, lr: float = 3e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    @autocast()
    def forward(self, batch: Dict[str, torch.Tensor]) -> FairnessOutput:
        x = batch["x"]  # (B,L,d)
        attn_mask = batch.get("attn_mask")

        x = self.projector(x)
        x = self.moe(x)

        # Attribute-specific adapters (utility restoration).
        for adapter in self.attr_adapters:
            x = x + adapter(x)

        task_logits = self.task_head(x)

        # Gradient reversal makes the shared representation reduce sensitive leakage.
        x_adv = grad_reverse(x, self.fairness_grl_scale)
        sensitive_logits = self.sensitive_head(x_adv)

        return FairnessOutput(task_logits=task_logits, sensitive_logits=sensitive_logits)


def fairness_loss(
    out: FairnessOutput,
    y: torch.Tensor,
    sensitive: torch.Tensor,
    fairness_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Ignore padded positions marked by -100.
    task_loss = F.cross_entropy(out.task_logits.reshape(-1, out.task_logits.shape[-1]), y.view(-1), ignore_index=-100)
    leakage_loss = F.cross_entropy(
        out.sensitive_logits.reshape(-1, out.sensitive_logits.shape[-1]),
        sensitive.view(-1),
        ignore_index=-100,
    )

    total = task_loss + fairness_weight * leakage_loss
    return total, {"task_loss": float(task_loss.detach()), "leakage_loss": float(leakage_loss.detach())}
