from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    pad_id: int = 0
    unk_id: int = 1


@dataclass(frozen=True)
class ToyConfig:
    vocab_size: int = 4000
    max_len: int = 32
    n_pool: int = 2000
    n_query: int = 64
    toxic_rate: float = 0.08

    # stage-1 decoder
    d_model: int = 128

    # LoRA
    lora_r: int = 8
    lora_alpha: float = 16.0

    # CountSketch
    sketch_dim: int = 256

    # stage-2 encoder
    enc_dim: int = 128


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # LoRA params
        self.lora_a = nn.Parameter(torch.empty(self.r, in_features))
        self.lora_b = nn.Parameter(torch.empty(out_features, self.r))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base
        y = F.linear(x, self.weight, self.bias)
        # LoRA branch
        delta = (x @ self.lora_a.t()) @ self.lora_b.t()
        return y + self.scaling * delta


class ToyDecoderClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_id: int, lora_r: int, lora_alpha: float) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.cls = LoRALinear(d_model, 2, r=lora_r, alpha=lora_alpha)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        denom = attn_mask.sum(dim=1).clamp(min=1).unsqueeze(1)
        pooled = (h * attn_mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.cls(pooled)

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield self.cls.lora_a
        yield self.cls.lora_b


class ToyEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_id: int, out_dim: int) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        denom = attn_mask.sum(dim=1).clamp(min=1).unsqueeze(1)
        pooled = (h * attn_mask.unsqueeze(-1)).sum(dim=1) / denom
        z = self.proj(pooled)
        return F.normalize(z, dim=-1)


class CountSketch:
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0) -> None:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.h = torch.randint(low=0, high=self.out_dim, size=(self.in_dim,), generator=g)
        self.s = torch.randint(low=0, high=2, size=(self.in_dim,), generator=g) * 2 - 1

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        # v: [in_dim]
        out = torch.zeros(self.out_dim, device=v.device, dtype=v.dtype)
        out.index_add_(0, self.h.to(v.device), self.s.to(v.device) * v)
        return out


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)


def flatten_grads(grads: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads], dim=0)


def per_example_lora_grad_sketch(
    model: ToyDecoderClassifier,
    x: torch.Tensor,
    attn_mask: torch.Tensor,
    y: torch.Tensor,
    sketch: CountSketch,
) -> torch.Tensor:
    # compute per-sample gradient for LoRA params
    logits = model(x.unsqueeze(0), attn_mask.unsqueeze(0))
    loss = F.cross_entropy(logits, y.unsqueeze(0))

    grads = torch.autograd.grad(loss, list(model.lora_parameters()), retain_graph=False, create_graph=False)
    gvec = flatten_grads([g.detach() for g in grads])
    sk = sketch(gvec)
    return F.normalize(sk, dim=0)


def spearmanr(x: torch.Tensor, y: torch.Tensor) -> float:
    # minimal spearman: rank correlation via argsort
    x = x.detach().cpu()
    y = y.detach().cpu()
    rx = torch.argsort(torch.argsort(x))
    ry = torch.argsort(torch.argsort(y))
    rx = rx.float() - rx.float().mean()
    ry = ry.float() - ry.float().mean()
    denom = (rx.norm() * ry.norm()).clamp(min=1e-8)
    return float((rx * ry).sum() / denom)


def auprc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    # y_true: {0,1}, y_score: higher means more positive
    y_true = y_true.detach().cpu().float()
    y_score = y_score.detach().cpu().float()

    order = torch.argsort(y_score, descending=True)
    y_true = y_true[order]

    tp = torch.cumsum(y_true, dim=0)
    fp = torch.cumsum(1 - y_true, dim=0)
    precision = tp / (tp + fp + 1e-8)

    # AP = sum_{k: y_k=1} precision@k / (#pos)
    pos = int(y_true.sum().item())
    if pos == 0:
        return 0.0
    ap = float((precision * y_true).sum().item() / pos)
    return ap
