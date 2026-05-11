"""
Valley3 MoE LLM backbone — standalone module.
See valley3.py for integrated implementation.

This module provides the MoE components separately for clarity:
  - MoERouter: top-k gating
  - ExpertFFN: individual expert (SwiGLU)
  - MoEBlock: full MoE FFN layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MoERouter(nn.Module):
    """
    Top-k router for Mixture-of-Experts.
    Uses simple linear gating with softmax normalization.

    Formula: g(x) = TopK(Softmax(W_g @ x))
    """

    def __init__(self, hidden_size: int, num_experts: int, k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch * seq_len, hidden_size]
        Returns:
            weights: [batch * seq_len, k] — normalized routing weights
            indices: [batch * seq_len, k] — selected expert indices
        """
        logits = self.gate(x)  # [N, num_experts]
        weights, indices = torch.topk(logits, self.k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return weights, indices


class ExpertFFN(nn.Module):
    """Single expert FFN using SwiGLU (same as LLaMA/Mistral)."""

    def __init__(self, hidden_size: int, ffn_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, ffn_size, bias=False)
        self.w2 = nn.Linear(ffn_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, ffn_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class MoEBlock(nn.Module):
    """
    Full MoE FFN block with top-k routing.
    Replaces standard FFN in transformer layers.
    """

    def __init__(self, hidden_size: int, num_experts: int, k: int = 2):
        super().__init__()
        self.router = MoERouter(hidden_size, num_experts, k)
        ffn_size = hidden_size * 4
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_size, ffn_size) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        x_flat = x.view(-1, H)

        weights, indices = self.router(x_flat)

        output = torch.zeros_like(x_flat)
        for k_idx in range(indices.shape[1]):
            expert_idx = indices[:, k_idx]
            weight = weights[:, k_idx:k_idx+1]
            for e in range(len(self.experts)):
                mask = (expert_idx == e)
                if mask.any():
                    output[mask] += weight[mask] * self.experts[e](x_flat[mask])

        return output.view(B, T, H)
