from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (64, 64), out_dim: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PropensityNet(nn.Module):
    """估计 e(x)=P(S=synthetic|x) 的域分类器（propensity）。"""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.mlp = MLP(d, hidden=(64, 64), out_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)  # logits


class OutcomeNet(nn.Module):
    """预测 p(y=1|x,t) 的 outcome 模型（在本 toy 里主要用 synthetic 训练）。"""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.mlp = MLP(d + 1, hidden=(128, 64), out_dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tt = t.float().unsqueeze(-1)
        inp = torch.cat([x, tt], dim=-1)
        return self.mlp(inp).squeeze(-1)  # logits

    def prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, t))


class BiasNet(nn.Module):
    """concept shift 校正项：输出 *logit* 上的残差修正。"""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.mlp = MLP(d + 1, hidden=(64, 64), out_dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tt = t.float().unsqueeze(-1)
        inp = torch.cat([x, tt], dim=-1)
        raw = self.mlp(inp).squeeze(-1)
        # 限幅（logit space），避免不稳定
        return 1.5 * torch.tanh(raw)


@dataclass
class DRModels:
    propensity: PropensityNet
    outcome: OutcomeNet
    bias: BiasNet


def bce_from_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y.float())


@torch.no_grad()
def auc_roc_binary(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """纯 torch 的 ROC-AUC（y_true in {0,1}）。"""

    y_true = y_true.detach().float().flatten()
    y_score = y_score.detach().float().flatten()
    # rank-based AUC: (sum ranks of positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    order = torch.argsort(y_score)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, order.numel() + 1, device=y_score.device, dtype=torch.float)

    n_pos = y_true.sum().clamp_min(1.0)
    n_neg = (1.0 - y_true).sum().clamp_min(1.0)
    sum_ranks_pos = (ranks * y_true).sum()

    auc = (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc.clamp(0.0, 1.0).cpu())


@torch.no_grad()
def effective_sample_size(w: torch.Tensor) -> float:
    w = w.detach().float().flatten().clamp_min(1e-12)
    return float((w.sum() ** 2 / (w.pow(2).sum().clamp_min(1e-12))).cpu())


@torch.no_grad()
def corrected_prob(outcome: OutcomeNet, bias: BiasNet, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    logits = outcome(x, t) + bias(x, t)
    return torch.sigmoid(logits)
